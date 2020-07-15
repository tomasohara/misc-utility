#! /usr/bin/env python
#
# Module to support coarse-grained parallelism, such as for when map-reduce via
# MRJob is overkill or awkward to program.
#
# TODO:
# - Piggyback on Chef processing (e.g., create sandbox role).
# - Use os.path.join, etc. throughout for transparent path handling.
# - Add package support (either via chef role or homegrown package install).
# - Add user-specific adhoc settings (e.g., bon.tar.gz under /root/bin with 
#   /root/bin added to path).
# - Add option too return archive of temporary worker directory.
#

"""
Provides RemoteDispatcher class for facilitating remote delegation of script
processing to client machines
"""

import os
import re
import tempfile
import time

from main import Main
import tpo_common as tpo
import glue_helpers as gh

SIMPLE_TEST = "simple-test"
RUN_COMMAND = "run-command"
JOB_NAME = "job-name"

HOME = tpo.getenv_text("HOME", "", "User's home directory")
JUJU_USER = tpo.getenv_text("JUJU_USER", "tohara", "User name for sandbox")
JUJU_RSA_FILE = tpo.getenv_text("JUJU_RSA_FILE", HOME + "/.ssh/juju-id_rsa", 
                                "The secret key for SSH access")
DEFAULT_REMOTE_WORKERS = """
    ec2-54-82-11-252.compute-1.amazonaws.com     ec2-107-21-145-66.compute-1.amazonaws.com 
    ec2-54-160-227-195.compute-1.amazonaws.com   ec2-54-167-97-22.compute-1.amazonaws.com  
    ec2-54-146-231-204.compute-1.amazonaws.com
"""
REMOTE_WORKERS = tpo.getenv_text("REMOTE_WORKERS", DEFAULT_REMOTE_WORKERS,
                                 "List of public IP addresses for worker hosts")
SKIP_SETUP = tpo.getenv_boolean("SKIP_SETUP", False, "Don't setup remote host")
FORCE_SETUP = tpo.getenv_boolean("FORCE_SETUP", False, "Always setup remote host")
UPDATE_SETUP = tpo.getenv_boolean("UPDATE_SETUP", False, "Update remote host setup")

#-------------------------------------------------------------------------------

def escape_ssh_command(command):
    """Ecapes characters in SSH CND such as single quote"""
    ## ex: escape_ssh_command("echo 'fubar'") => "echo \'fubar\'"
    escape_command = command.replace("'", "\'")
    tpo.debug_format("escape_ssh_command{{cmd}) => {esc}", 7,
                     cmd=command, esc=escape_command)
    return escape_command

#------------------------------------------------------------------------

class RemoteDispatcher(object):
    """Class for faciliating course-grained parallelism via remote clients
    Note: worker_num parameter used in methods is 0-based"""

    def __init__(self, remote_workers=None, all_sandbox=False,
                 job_name=None, temp_dir=None, remote_temp_dir=None, skip_setup=SKIP_SETUP):
        # TODO: add support for required packages
        """Class constructor, initializing list of REMOTE_WORKERS"""
        tpo.debug_format("RemoteDispatcher.__init__({rw}, {all}, {jn}, {td}, {rtd})", 5,
                         rw=remote_workers, all=all_sandbox, jn=job_name,
                         td=temp_dir, rtd=remote_temp_dir)
        if not job_name:
            job_name = "adhoc-job"
        self.job_name = job_name
        if not remote_workers:
            remote_workers = REMOTE_WORKERS.replace(",", " ").split()
        self.remote_workers = remote_workers
        if not remote_temp_dir:
            remote_temp_dir = os.path.join("/tmp", self.job_name)
        self.remote_temp_dir = remote_temp_dir
        if not temp_dir:
            temp_dir = tpo.getenv_text("TEMP_BASE",
                                       tempfile.NamedTemporaryFile().name)
        self.temp_dir = temp_dir
        self.skip_setup = skip_setup
        self.juju_rsa_file_path = None
        self.ssh_options = ""
        self.do_prejob_setup(all_sandbox)
        return

    def do_prejob_setup(self, all_sandbox):
        """Perform setup steps prior to running jobs"""
        tpo.debug_format("RemoteDispatcher.do_prejob_setup({all})", 4, 
                         all=all_sandbox)

        # Initialize SSH stuff
        self.juju_rsa_file_path = gh.resolve_path(JUJU_RSA_FILE)
        gh.assertion(gh.non_empty_file(self.juju_rsa_file_path))
        self.ssh_options = "-o StrictHostKeyChecking=no"

        # Make sure job-specific temp directories exists (local and remote)
        gh.run("mkdir -p {temp}", temp=self.temp_dir)
        self.run_command_on_workers("mkdir -p {temp}",
                                    temp=self.remote_temp_dir)
        # Do other host setup steps
        if (not self.skip_setup):
            self.setup_remote_workers(all_sandbox)
        return

    def copy_file_to_worker(self, worker_num, filename, dest_dir=None):
        """Uploads FILENAME to host for WORKER_NUM under DEST_DIR (or job's temp dir)"""
        tpo.debug_format("RemoteDispatcher.copy_file_to_worker({n}, {f}, {d})", 4,
                         n=worker_num, f=filename, d=dest_dir)
        if dest_dir is None:
            dest_dir = self.temp_dir
        host = self.remote_workers[worker_num]
        out = gh.run("scp {opt} -i {rsa} {f} root@{h}:{tmp}", 5,
                     opt=self.ssh_options, rsa=self.juju_rsa_file_path, f=filename, 
                     h=host, tmp=self.remote_temp_dir)
        return out

    def copy_file_from_worker(self, worker_num, filename, dest_dir=None):
        """Downloads FILENAME from host for WORKER_NUM to current DEST_DIR"""
        tpo.debug_format("RemoteDispatcher.copy_file_to_worker({n}, {f}, {d})", 4,
                         n=worker_num, f=filename, d=dest_dir)
        if dest_dir is None:
            dest_dir = self.temp_dir
        host = self.remote_workers[worker_num]
        out = gh.run("scp {opt} -i {rsa} root@{h}:{f} {dir}", 5,
                     opt=self.ssh_options, rsa=self.juju_rsa_file_path, f=filename, 
                     h=host, dir=dest_dir)
        return out

    def copy_file_to_workers(self, filename, dest_dir=None):
        """Uploads FILENAME to each of the workers under DEST_DIR.
        Note: destination defaults is job's temp dir (e.g.. /tmp/count-hits)"""
        tpo.debug_format("RemoteDispatcher.copy_file_to_workers({f}, {d})", 4,
                         f=filename, d=dest_dir)
        if dest_dir is None:
            dest_dir = self.remote_temp_dir
        out = ""
        for host in self.remote_workers:
            out += gh.run("scp {opt} -i {rsa} {f} root@{h}:{dir}", 5,
                          opt=self.ssh_options, rsa=self.juju_rsa_file_path, 
                          f=filename, h=host, dir=dest_dir)
        return out

    def read_worker_file(self, num, file_path):
        """Reads file at PATH from worker NUM"""
        self.copy_file_from_worker(num, file_path)
        file_name = gh.basename(file_path)
        file_contents = gh.read_file(os.path.join(self.temp_dir, file_name))
        return file_contents

    def archive_directory(self, dirname):
        """Creates compressed archive of DIRNAME under temp dir"""
        tpo.debug_format("RemoteDispatcher.archive_directory({d})", 4,
                         d=dirname)
        basename = gh.basename(dirname)
        archive_path = tpo.format("{temp}/{base}.tar.gz", 
                                  temp=self.temp_dir, base=basename)
        verbose = "v" if tpo.verbose_debugging() else ""
        gh.run("tar cfz{v} {arc} {dir}",
               arc=archive_path, dir=dirname, v=verbose)
        return archive_path

    def extract_archive_on_workers(self, archive_path, dest_dir=None):
        """Copies ARCHIVE_PATH to workers and extracts under DEST_DIR"""
        tpo.debug_format("RemoteDispatcher.extract_archive_on_workers({arc})",
                         4, arc=archive_path)
        if dest_dir is None:
            dest_dir = self.remote_temp_dir
        self.copy_file_to_workers(archive_path, self.remote_temp_dir)
        # Extract on each
        verbose = "v" if tpo.verbose_debugging() else ""
        archive_basename = gh.basename(archive_path)
        out = self.run_command_on_workers("cd {dest} && tar xfz{v} {tmp}/{arc}", 
                                          v=verbose, tmp=self.remote_temp_dir,
                                          dest=dest_dir, arc=archive_basename)
        return out

    def copy_dir_to_workers(self, dirname):
        """Copies directory DIRNAME to each of the workers under remote temp"""
        tpo.debug_format("RemoteDispatcher.copy_dir_to_workers({d})", 4,
                         d=dirname)
        # Create zipped tar archive
        archive_path = self.archive_directory(dirname)
        # Copy over on each
        out = self.extract_archive_on_workers(archive_path)
        return out

    def run_command_on_worker(self, worker_num, command, **namespace):
        """Run COMMAND on host for WORKER_NUM"""
        tpo.debug_format("RemoteDispatcher.run_command_on_worker({n}, {cmd}, {ns})",
                         5, n=worker_num, cmd=command, ns=namespace)
        # note: Runs with /root as current directory on worker
        host = self.remote_workers[worker_num]
        full_command = command
        if re.search("{.*}", full_command):
            full_command = escape_ssh_command(tpo.format(full_command, **namespace))
        out = gh.run("ssh {opt} -i {rsa} root@{h} '{cmd}'", 5,
                     opt=self.ssh_options, rsa=self.juju_rsa_file_path, 
                     h=host, cmd=full_command)
        return out

    def run_command_on_workers(self, command, **namespace):
        """Run COMMAND on each of WORKERS"""
        # note: Runs with /root as current directory on worker
        tpo.debug_format("RemoteDispatcher.run_command_on_workers({cmd}, {ns})", 6,
                         cmd=command, ns=namespace)
        full_command = command
        if re.search("{.*}", full_command):
            full_command = escape_ssh_command(tpo.format(full_command, **namespace))
        out = ""
        for host in self.remote_workers:
            out += gh.run("ssh {opt} -i {rsa} root@{h} '{cmd}'", 5,
                          opt=self.ssh_options, rsa=self.juju_rsa_file_path, 
                          h=host, cmd=full_command)
        return out
    
    def kill_process_on_workers(self, process_pattern):
        """Kills process(es) for PROCESS_PATTERN on each of the remote workers"""
        tpo.debug_format("RemoteDispatcher.kill_process_on_workers({pat})", 6,
                         pat=process_pattern)
        out = self.run_command_on_workers(tpo.format("/root/kill_em.sh -p {pat}",
                                                     pat=process_pattern))
        return out

    def setup_entire_sandbox(self, force=FORCE_SETUP, update=UPDATE_SETUP):
        """Make sure Juju sandbox repository setup (via Mercurial)"""
        src_dir = "/var/juju/src"
        hg_branch = self.read_worker_file(0, src_dir + "/sandbox/.hg/branch")
        if force or not hg_branch:
            # Make sure read-only acess
            # TODO: make a copy of existing .hgrc
            hg_settings = self.read_worker_file(0, "/root/.hgrc")
            hg_settings_file = os.path.join(self.temp_dir, ".hgrc")
            hg_settings += ("\n" +
                            "[auth]\n" +
                            "repo.prefix = https://hg.juju.com/hg\n" +
                            "repo.username = hgmaster\n" +
                            "repo.password = T67g32nL\n" +
                            "[hostfingerprints]\n" +
                            "hg.juju.com = 7f:20:2c:99:4d:3d:ac:fd:3d:7d:e0:61:c2:2c:29:a1:c3:de:df:81")
            gh.write_file(hg_settings_file, hg_settings)
            self.copy_file_to_workers(hg_settings_file, dest_dir="/root")

            # Check-out the sandbox
            cmd = ("mkdir -p {src} && cd {src} &&" +
                   " hg clone 'https://hg.juju.com/hg/sandbox'")
            self.run_command_on_workers(cmd, src=src_dir)
        elif update:
            self.run_command_on_workers("cd {src}/sandbox && hg pull && hg update", 
                                        src=src_dir)
            
        # Make sure sandbox for user is in Unix path and python path
        # TODO: make copy of existing .bashrc
        bash_settings = self.read_worker_file(0, "/root/.bashrc")
        if not "sandbox" in bash_settings:
            sandbox_dir = src_dir + "/sandbox/" + JUJU_USER
            bash_settings_file = os.path.join(self.temp_dir, ".bashrc")
            bash_settings += "\n" + tpo.format("export PATH=$PATH:{sandbox}\n"
                                               + "export PYTHONPATH=$PYTHONPATH:{sandbox}",
                                               sandbox=sandbox_dir)
            gh.write_file(bash_settings_file, bash_settings)
            self.copy_file_to_workers(bash_settings_file, dest_dir="/root")
        return

    def setup_remote_workers(self, all_sandbox=False, archives=None):
        """Setup each of the REMOTE_WORKERS to act as adhoc clients"""
        if not archives:
            archives = []
        # Copy the package setup script to remote hosts and run
        tpo.debug_format("RemoteDispatcher.setup_remote_workers([all])", 6,
                         all=all_sandbox)
        setup_script = os.path.join(self.temp_dir, "setup_remote_workers.sh")
        # note: csh used for supporting scripts (e.g., kill_em.sh)
        gh.write_lines(setup_script, 
                       ["apt-get --yes install tcsh"])
        self.copy_file_to_workers(setup_script)
        self.run_command_on_workers("source {script}", script=setup_script)

        # Copy sandbox scripts to remote hosts and extract under temp dir
        ## OLD: self.extract_archive_on_workers("/var/juju/pkgs/tohara-1.1.tar.gz")
        for archive in archives:
            self.extract_archive_on_workers(archive)
        if all_sandbox:
            self.setup_entire_sandbox()
        # Copy kill_em.sh to remote hosts for convenient worker termination.
        # TODO: rework kill script in python to avoid tcsh installation
        kill_script = gh.resolve_path("kill_em.sh")
        self.copy_file_to_workers(kill_script, "/root")
        return

    def start_remote_interface(self):
        """Start supplemental interface on each of the remote workers"""
        tpo.debug_format("RemoteDispatcher.start_remote_interface(); *** stub ***: self={s}", 5,
                         s=self)
        gh.assertion(False)
        return

    def stop_remote_interface(self):
        """Stop supplemental interface on each of the remote workers"""
        tpo.debug_format("RemoteDispatcher.stop_remote_interface(); *** stub ***: self={s}", 5,
                         s=self)
        gh.assertion(False)
        return

    def get_bash_environment_options(self, ignore=None):
        """Return environnment option settings formatted in bash command-prefix style (see register_env_option in tpo_common)"""
        tpo.debug_format("RemoteDispatcher.get_bash_environment_options({i}): self={s}", 6,
                         i=ignore, s=self)
        ignore_list = ignore.split()
        environment_spec = ""
        for var in tpo.get_registered_env_options():
            if var in ignore_list:
                tpo.debug_format("get_bash_environment_options: Ignoring variable '{v}'", 
                                 7, v=var)
                continue
            value = tpo.getenv(var)
            if environment_spec:
                environment_spec += " "
            environment_spec += (var + "=" + value)
        tpo.debug_format("RemoteDispatcher.get_bash_environment_options() => '{spec}'", 
                         5, spec=environment_spec)
        return (environment_spec)

#------------------------------------------------------------------------

class Script(Main):
    """Input processing class"""
    simple_test = None
    command = None
    dispatcher = None
    num_workers = 0
    job_name = "remote-job"

    def setup(self):
        """Check results of command line processing"""
        tpo.debug_format("Script.setup(): self={s}", 5, s=self)
        self.simple_test = self.get_parsed_option(SIMPLE_TEST)
        self.command = self.get_parsed_option(RUN_COMMAND, "")
        self.job_name = self.get_parsed_option(JOB_NAME, self.job_name)
        tpo.trace_object(self, label="Script instance")

    def run_main_step(self):
        """Main processing step"""
        tpo.debug_format("Script.run_main_step(): self={s}", 5, s=self)
        if self.simple_test:
            self.run_simple_test()
        elif self.command:
            self.run_command_remotely()
        else:
            self.parser.print_help()

    def run_simple_test(self):
        """Runs simple test of remote dispatching"""
        tpo.debug_format("Script.run_simple_test(): self={s}", 5, s=self)
        # Trace environment
        print("Environment options:")
        print("\t" + tpo.formatted_environment_option_descriptions())
        #
        print("Running sleep in background on clients")
        dispatcher = RemoteDispatcher(job_name="simple-test")
        dispatcher.run_command_on_workers("ps auxg | grep python")
        dispatcher.run_command_on_workers("echo 'sleep 360 &' >| {tmp}/sleep.sh && source /tmp/sleep.sh", 
                                          tmp=dispatcher.remote_temp_dir)
        #
        print("Copying script to clients")
        dispatcher.copy_file_to_workers(__file__)
        #
        print("Killing our sleep process on clients")
        dispatcher.kill_process_on_workers("sleep.360")
        #
        print("Copying python scripts to clients")
        xfer_dir = os.path.join(dispatcher.temp_dir, "python-scripts")
        gh.run("mkdir {dir} && cp -p *.py {dir}", dir=xfer_dir)
        dispatcher.copy_dir_to_workers(xfer_dir)
        return

    def run_command_remotely(self):
        """Partition input and run remotely
        Note: command is assumed to accept filename parameter for input 
        and it should not redirect final output"""
        # Make sure no output and log direction
        # exs: "echo fubar > fubar.txt" and "echo fubar 1>| fubar.txt 2>| fubar.log"
        gh.assertion(not re.search(r" [12]?>\|? ", self.command))
        # TODO: move most of support into RemoteDispatcher
        self.dispatcher = RemoteDispatcher(job_name="split-test")
        self.num_workers = len(self.dispatcher.remote_workers)
        # Split input into pieces, including headers in each
        file_basename = gh.basename(self.filename)
        flag_file = tpo.format("{tmp}/process-{b}.done",
                               b=file_basename, tmp=self.dispatcher.remote_temp_dir)
        output = gh.run("python -m split_file --include-header --output-base {b} --num-splits {n} {f} 2>| {l}",
                        n=self.num_workers, b=os.path.join(self.dispatcher.temp_dir, "input"),
                        f=self.filename, l=os.path.join(self.dispatcher.temp_dir, "split.log"))
        split_files = output.split()
        output_file_template = os.path.join(self.dispatcher.remote_temp_dir, "worker-{w}.output")
        # Send pieces to each worker and start each remote process
        for w in range(self.num_workers):
            split_file = split_files[w]
            self.dispatcher.copy_file_to_worker(w, split_file)
            split_basename = gh.basename(split_file)
            command_script = tpo.format("{tmp}/process-{b}.sh",
                                        b=split_basename, tmp=self.dispatcher.temp_dir)
            # Create main script for commands to run
            output_file = tpo.format(output_file_template, w=w)
            log_file = output_file + ".log"
            gh.write_lines(command_script, 
                           [## TODO: tpo.format("cd {tmp}", tmp=self.dispatcher.remote_temp_dir),
                            # HACK: force the .bashrc to be source (TODO, put PYTHONPATH elsewhere)
                            tpo.format("PS1=fubar && source /root/.bashrc"),
                            tpo.format("rm -f {ff}", ff=flag_file),
                            tpo.format("{cmd} {tmp}/{b} >| {out} 2>| {log}", 
                                       cmd=self.command, tmp=self.dispatcher.remote_temp_dir, 
                                       b=split_basename, out=output_file, log=log_file),
                            tpo.format("touch {ff}", ff=flag_file)])
            # Create script to run commands in backgrund
            invocation_script = tpo.format("{tmp}/invoke-process-{b}.sh",
                                           b=split_basename, tmp=self.dispatcher.temp_dir)
            gh.write_lines(invocation_script, 
                           [tpo.format("source {tmp}/{scr} &",
                                       tmp=self.dispatcher.remote_temp_dir,
                                       scr=gh.basename(command_script))])
            self.dispatcher.copy_file_to_worker(w, command_script)
            self.dispatcher.copy_file_to_worker(w, invocation_script)
            self.dispatcher.run_command_on_worker(w, "source {tmp}/{scr}",
                                                  tmp=self.dispatcher.remote_temp_dir,
                                                  scr=gh.basename(invocation_script))
        # Wait for jobs to finish
        self.wait_for_finish(flag_file, output_file_template)

    def wait_for_finish(self, flag_file, output_file_template):
        """Wait for each worker to finish or for timeout"""
        # TODO: into RemoteDispatcher
        gh.assertion("{w}" in output_file_template)
        # TODO: record process ID and make sure still running remotely
        MAX_SLEEP = tpo.getenv_integer("MAX_SLEEP", 86400, "Maximum seconds to wait for finish")
        SLEEP_SECONDS = tpo.getenv_integer("SLEEP_SECONDS", 60, "Number of seconds to wait during end polling")
        num_left = self.num_workers
        still_running = [True] * self.num_workers
        time_slept = 0
        while num_left > 0:
            if time_slept >= MAX_SLEEP:
                tpo.print_stderr(tpo.format("Error: time out reached ({MAX_SLEEP} seconds) in invoke_distributed"))
                break
            time.sleep(SLEEP_SECONDS)
            time_slept += SLEEP_SECONDS
 
            # Check each active host for completion, downloading results when reached.
            for w in range(self.num_workers):
                if still_running[w]:
                    command = tpo.format("ls {ff} 2> /dev/null", ff=flag_file)
                    flag_found = self.dispatcher.run_command_on_worker(w, command)
                    if flag_found:
                        still_running[w] = False
                        num_left -= 1
                        remote_output_file = tpo.format(output_file_template,
                                                        w=w)
                        self.dispatcher.copy_file_from_worker(w, remote_output_file)
                        local_output_file = os.path.join(self.dispatcher.temp_dir,
                                                         gh.basename(remote_output_file))
                        tpo.debug_format("Output from host {w}: {out}", 5,
                                         w=w, out=local_output_file)
                        print(gh.read_file(local_output_file))
                        if tpo.detailed_debugging():
                            self.dispatcher.copy_file_from_worker(w, remote_output_file + ".log")
                            tpo.debug_format("Log from host {w} {name}: {{\n{log}\n\t}}\n", 1,
                                             w=w, name=(local_output_file + ".log"),
                                             log=gh.indent_lines(gh.read_file(local_output_file + ".log")))
        return

#------------------------------------------------------------------------

if __name__ == '__main__':
    tpo.trace_current_context(level=tpo.QUITE_DETAILED)
    app = Script(description=__doc__,
                 skip_input=True,
                 boolean_options=[SIMPLE_TEST],
                 text_options=[(RUN_COMMAND, "Run given command remotely over partitioned input"),
                               (JOB_NAME, "Name for job (e.g., to keep files temp files separate", Script.job_name)])
    app.run()
