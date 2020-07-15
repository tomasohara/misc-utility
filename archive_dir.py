#! /usr/bin/env python
#
# archive_dir.py: packages a directory into chunks and xfers to another location
# such as another host or Amazon S3 storage. This is useful when there are size
# restrictions on the destination (as with S3 uploads) or when the local host doesn't
# have enough disk space to create a large tar file (e.g., prior to split).
# It also can be used to create more natural splits (e.g., at subdir level rather than
# byte or line).
#
# Note:
# - For simplicity chunking occurs on subdirectory boundaries, so it still can
#   produce archives larger than the chunk size.
# - Assumes running under Unix (e.g., for use of du for file usage,
#   and scp for remote copying).
# - S3 support requires s3cmd package.
#
# TODO:
# - Rename as package_and_ship.py??
# - Add support for Unix pipelining (e.g., 'tar dir | split | scp').
# - Add sanity check for collision of derived file paths and actual ones.
# - Add exception handling to recovery from unexpected errors (e.g., lack
#   of free space due to excessively large archive files).
# - Add option for maximum depth for separate subdir archives (to avoid
#   having too many small ones).
# - Add options for max archive files when not using separate subdirs
#   (to avoid too many large archives).
# - ** Add option for max time (i.e., incremental update).
# - Add option to disable splitting given multipart usage (--multipart-chunk-size-mb=4096).
#

"""Packages a directory into chunks and transfers elsewhere"""

import argparse
import os
## TODO: import re
import sys
import tempfile

import tpo_common as tpo
import glue_helpers as gh


MAX_CHUNK_MB = tpo.getenv_number("MAX_CHUNK_MB", 4096,
                                 "Max size of archive chunk in MB")
SEPARATE_SUBDIRS = tpo.getenv_boolean("SEPARATE_SUBDIRS", False,
                                      "Use separate archive per subdirectory")
ARCHIVE_NAME = tpo.getenv_text("ARCHIVE_NAME", "archive",
                               "Base name for archive files")
# TODO: make ADD_SUBDIR_AFFIX True by default if SEPARATE_SUBDIRS enabled
ADD_SUBDIR_AFFIX = tpo.getenv_boolean("ADD_SUBDIR_AFFIX", False,
                                      "Include subdirectory in archive names")
TEMP_DIR = tpo.getenv_text("TMP", "/tmp", "Temporary directory")
AFFIX_DELIM = tpo.getenv_text("AFFIX_DELIM", "__",
                              "Text to separate affixes in derived filenames")
META_PREFIX = tpo.getenv_text("META_CHAR", "@",
                              "Prefix for meta affixes (e.g., for split affix)")
NUMBER_ARCHIVES = tpo.getenv_boolean("NUMBER_ARCHIVES", not ADD_SUBDIR_AFFIX,
                                     "Add index number for archive to basename")
EST_COMPRESSION_FACTOR = tpo.getenv_number("EST_COMPRESSION_FACTOR", 0.5,
                                           "Estimated compression factor")
VERBOSE = tpo.getenv_boolean("VERBOSE", False, "Verbose output mode")

def subdir_usage(dirname):
    """Derives hash of sub-directories for DIR and the usage of each under
    the file system (in bytes).
    Note: Includes usage the directory itself via . placeholder."""

    # Derive usage similar to `du --one-file-system` but with fields reversed
    # and convert into hash.
    # note: dereference symlinks (via --dereference-args)
    dir_usage = {}
    du_listing = gh.run("du --dereference-args --one-file-system '{dir}' 2>&1",
                        dir=dirname, trace_level=5)
    for line_num, subdir_info in enumerate(du_listing.split("\n")):
        try:
            (size, subdir) = subdir_info.split("\t")
            dir_usage[subdir] = tpo.safe_int(size)
        except ValueError:
            tpo.debug_format("Problem extracting du info at line {l}: {info}",
                             3, l=line_num, info=subdir_info)
    tpo.debug_format("subdir_usage({d}) => {h}", 6, d=dirname, h=dir_usage)
    return dir_usage


class packager(object):
    """Class for archiving a directory into chuncks with optional xfer"""

    def __init__(self, dest, max_chunk_size=MAX_CHUNK_MB,
                 archive_name=ARCHIVE_NAME, temp_dir=TEMP_DIR):
        """Class constructor accepting the target destination"""
        tpo.debug_format("packager.__init__({d}, [{mcs}, {an}, {td}])", 5,
                         d=dest, mcs=max_chunk_size, an=archive_name,
                         td=temp_dir)
        self.dest = dest
        self.archive_name = archive_name
        self.archive_num = 0
        self.temp_dir = temp_dir

        # Derive the chunk size (n.b., S3 is limited to 5GB upload chunks)
        self.max_chunk_KB = max_chunk_size * 1024.0
        gh.assertion(self.max_chunk_KB > 0)
        KB = 1024
        MB = KB * KB
        GB = KB * MB
        max_chunk_bytes = self.max_chunk_KB * KB
        gh.assertion(max_chunk_bytes <= (64 * GB))
        if self.dest.startswith("s3:"):
            ## OLD assertion since --multipart-chunk-size-mb=4096 handles this
            ## gh.assertion(max_chunk_bytes <= (5 * GB))
            if not self.dest.endswith("/"):
                self.dest += "/"

        self.temp_file = tpo.getenv_text("TEMP_FILE",
                                         tempfile.NamedTemporaryFile().name)
        self.log_file = self.temp_file + ".log"
        return

    def archive_and_xfer(self, file_list, affix=None):
        """Create archive with FILE_LIST, optionally adding AFFIX to derived
        FILENAME. If resulting archive is too large, it is split into pieces"""
        # Note: workhorse routine for package_and_xfer
        # TODO: check for errors in shell command execution (e.g., disk space)
        tpo.debug_format("archive_and_xfer({fl}, {aff})", 4,
                         fl=file_list, aff=affix)
        self.archive_num += 1

        # Derive name for the archive
        # TODO: resolve quirks in the derived name (e.g., ____)
        basename = os.path.join(self.temp_dir, self.archive_name)
        if NUMBER_ARCHIVES:
            basename += str(self.archive_num)
        if affix:
            basename += AFFIX_DELIM + affix
        basename = basename.replace("____", "__")
        tar_file = basename + ".tar.gz"

        # Create the archive
        # options: c[reate], f[ile], z[ip], and v[erbose]
        tar_options = "--one-file-system --create --verbose --gzip --file"
        gh.run("tar {opts} '{tar}' {files} >> {log} 2>&1", trace_level=5,
               tar=tar_file, opts=tar_options, log=self.log_file,
               files=" ".join(['"' + f + '"' for f in file_list]))
        xfer_files = [tar_file]

        # Split into pieces if too large
        max_bytes = int(1024 * self.max_chunk_KB)
        if gh.file_size(tar_file) > max_bytes:
            prefix = tar_file + AFFIX_DELIM + META_PREFIX + "part"
            gh.run("split {opts} --numeric-suffixes --bytes={b} '{tar}' '{pre}'"
                   + " >> {log} 2>&1", opts="--verbose", trace_level=5,
                   tar=tar_file, b=max_bytes, log=self.log_file, pre=prefix)
            # TODO: derive list of files from split output
            # ex: "creating file `/tmp/_kivy-examples-mb.tar.gz__@par00' ..."
            xfer_files = gh.get_matching_files(prefix + "*")
        gh.assertion(len(xfer_files) > 0)

        # Transfer tar file(s) to destination
        # TODO: add sanity check via (remote) directory listing
        gh.write_lines(self.log_file, ["archive: " + tar_file] + file_list,
                       append=True)
        if self.dest.startswith("s3:"):
            # note: --multipart-chunk-size used because some archives can exceed max chunk size (e.g., root directories)
            command = "s3cmd put --verbose --multipart-chunk-size-mb=4096 {file_spec} '{dest}'"
        elif ":" in self.dest:
            command = "scp -v {file_spec} '{dest}'"
        else:
            command = "cp -v {file_spec} '{dest}'"
        command += " >> {log}"
        file_spec = " ".join(['"' + f + '"' for f in xfer_files])
        gh.run(command, file_spec=file_spec, dest=self.dest, log=self.log_file, trace_level=5)
        gh.write_lines(self.log_file, ["-" * 80], append=True)

        # Remove temporary archive file(s)
        if not tpo.verbose_debugging():
            for f in tpo.append_new(xfer_files, tar_file):
                gh.delete_file(f)
        return

    def package_and_xfer(self, dirname, usage_hash=None):
        """Packages DIR into archives and XFER.
        Note: sudirectories are recursively processed."""
        tpo.debug_format("package_and_xfer({d}, [{uh}])", 4,
                         d=dirname, uh=usage_hash)

        # Derive space requirements unless recursive invocation
        top_level = False
        if not usage_hash:
            usage_hash = subdir_usage(dirname)
            top_level = True

        # Get current space requirements
        gh.assertion(os.path.exists(dirname))
        space_required = usage_hash.get(dirname)
        if not space_required:
            space_required = usage_hash.get(".", 0)
            gh.assertion(top_level)
        gh.assertion(space_required > 0)
        space_required *= EST_COMPRESSION_FACTOR

        # Format optional directory-based affix
        affix = None
        if ADD_SUBDIR_AFFIX:
            gh.assertion(not dirname.startswith(".."))
            dir_label = dirname if (dirname != ".") else ""
            dir_label = dirname.replace("./", "")
            affix = dir_label.replace("/", AFFIX_DELIM)

        # If the entire directory within limits, archive it as is,
        # unless creating separate archives per subdirectory.
        if (not SEPARATE_SUBDIRS) and (space_required < self.max_chunk_KB):
            self.archive_and_xfer([dirname], affix)

        # Otherwise decompose into chunks
        else:
            def dir_path(file_name):
                """Resolves path for FILE_NAME, incorporating current directory.
                Note: Uses DIRNAME from context"""
                return os.path.join(dirname, file_name)

            # Get lists of regular files and subdirctories
            all_file_names = gh.get_directory_listing(dirname, make_unicode=True)
            subdir_paths = []
            file_paths = []
            for file_name in all_file_names:
                path = dir_path(file_name)
                if os.path.isdir(path) \
                        and not os.path.islink(path) \
                        and not os.path.ismount(path):
                    subdir_paths.append(path)
                elif os.path.isfile(path):
                    file_paths.append(path)
                else:
                    tpo.debug_format("Ignoring non-regular file {f}", 4, f=file_name)

            # Create single archive for files in current dir
            if file_paths:
                files_affix = None
                if affix:
                    files_affix = affix
                    if not SEPARATE_SUBDIRS:
                        # Use __files suffix (e.g., to distinguish from __part)
                        files_affix += AFFIX_DELIM + META_PREFIX + "files"
                self.archive_and_xfer(file_paths, files_affix)
            else:
                tpo.debug_format("No regular files in {dir}", 5, dir=dirname)

            # Create separate archive for each subdirectory
            for path in subdir_paths:
                self.package_and_xfer(path, usage_hash)
        return


def main():
    """Entry point for script"""
    tpo.debug_print("main(): sys.argv=%s" % sys.argv, 4)

    # Check command-line arguments
    # TODO: add in detailed usage notes w/ environment option descriptions
    env_options = tpo.formatted_environment_option_descriptions(indent="    ")
    notes = tpo.format("""
Note: The following environment options are available:
    {env}
""", env=env_options)
    parser = argparse.ArgumentParser(description=__doc__, epilog=notes,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # TODO: allow for multiple source diretories
    parser.add_argument("filename",
                        help="Source directory filename")
    parser.add_argument("destination",
                        help="Target destination (e.g., directory or S3 folder")
    args = vars(parser.parse_args())
    tpo.debug_print("args = %s" % args, 5)
    filename = args['filename']
    dest = args['destination']

    # Do the archiving and xfer
    pkg = packager(dest)
    pkg.package_and_xfer(filename)

    # Trace log (TODO, put this in packager class)
    if VERBOSE:
        tpo.debug_format("log contents: {{\n{log}\n}}", 4,
                         log=gh.indent_lines(gh.read_file(pkg.log_file, 
                                                          make_unicode=True)))

    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
