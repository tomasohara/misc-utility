#! /usr/bin/env python
# 
# Gets random assertions from wikidata by selecting entities and showing a certain numer
# of assertions per entity.
#
# Several different ways of filtering data are available, such as following environment options:
#   SIMPLE_NAMES: name is simple English token (e.g., no spaces)
#   INCLUDE_REFLEXIVE: include reflexive relations
#   ENTITY_FILENAME: list of entities to include
#   RELATION_FILENAME: list of relations to include
# The latter can be based on PageRank-style analysis of the WikiData network. See
#   https://github.com/athalhammer/danker and 
#
#--------------------------------------------------------------------------------
#
# The software is Open Source, licensed under the GNU Lesser General Public Version 3 (LGPLv3). See LICENSE.txt in repository.
# LD;TR: You get what you paid for, so don't sue me!
#

"""Display random Wikidata assertions"""

# Standard packages
import logging
import random
import re
import time

# Installed packages
import wikidata.client

# Local packages
import debug
from main import Main
import system

# Constants
ID_PREFIX = "Q"
LAST_ID = system.getenv_int("LAST_ID", 90000000,
                            "Last Q ID in WikiData")
MAX_ENTITIES = system.getenv_int("MAX_ENTITIES", 100,
                                 "Number of entities to include")
MAX_PROPS = system.getenv_int("MAX_PROPS", 10,
                              "Maxium number of properties for each entity")
MAX_ASSERTIONS = system.getenv_int("MAX_ASSERTIONS", (MAX_ENTITIES * MAX_PROPS),
                                   "Maximum number of assertions to include")
SEED = system.getenv_int("SEED", 15485863,
                         "Random seed (if non-zero; defaults to one-milltionth prime)")
DEFAULT_PAUSE_SECS = 0.25 if (debug.get_level() < 2) else 0.5
PAUSE_SECS = system.getenv_int("PAUSE_SECS", DEFAULT_PAUSE_SECS,
                               "Time to sleep after each download")
SIMPLE_NAMES = system.getenv_bool("SIMPLE_NAMES", False,
                                  "Require entity names to be single words")
INCLUDE_REFLEXIVE = system.getenv_bool("INCLUDE_REFLEXIVE", False,
                                       "Include symmetric assertions")
SKIP_REFLEXIVE = (not INCLUDE_REFLEXIVE)
ENTITY_FILENAME = system.getenv_text("ENTITY_FILENAME", "",
                                     "Filename for list of entities to include")
RELATION_FILENAME = system.getenv_text("RELATIONS_FILENAME", "",
                                       "Filename for list of relations to include")


## TODO: Constants for switches omitting leading dashes (e.g., DEBUG_MODE = "debug-mode")
## Note: Run following in Emacs to interactively replace TODO_ARGn with option label
##    M-: (query-replace-regexp "todo\\([-_]\\)argn" "arg\\1name")
## where M-: is the emacs keystroke short-cut for eval-expression.
TODO_ARG1 = "TODO-arg1"
## TODO_ARG2 = "TODO-arg2"
## TODO_FILENAME = "TODO-filename"

def is_simple_name(name):
    """Whether NAME is simple (e.g., single English token)"""
    simple = re.search("^[a-z]+$", name, re.IGNORECASE)
    debug.trace(5, f"is_simple_name({name}) => {simple}")
    return simple

class Script(Main):
    """Input processing class"""
    # TODO: -or-: """Adhoc script class (e.g., no I/O loop, just run calls)"""
    ## TODO: class-level member variables for arguments (avoids need for class constructor)
    TODO_arg1 = False
    ## TODO_arg2 = ""
    entity_hash = None
    relation_hash = None

    def setup(self):
        """Check results of command line processing"""
        debug.trace(5, f"Script.setup(): self={self}")
        ## TODO: extract argument values
        self.TODO_arg1 = self.get_parsed_option(TODO_ARG1, self.TODO_arg1)
        ## self.TODO_arg2 = self.get_parsed_option(TODO_ARG2, self.TODO_arg2)
        # TODO: self.TODO_filename = self.get_parsed_argument(TODO_FILENAME)
        if SEED:
            random.seed(SEED)
        if ENTITY_FILENAME:
            self.entity_hash = system.create_boolean_lookup_table(ENTITY_FILENAME)
        if RELATION_FILENAME:
            self.relation_hash = system.create_boolean_lookup_table(RELATION_FILENAME)

        # Enable logging if debugging
        if (debug.debugging()):
            level = (logging.INFO if (debug.get_level() < debug.QUITE_DETAILED) else logging.DEBUG)
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=level)

        debug.trace_object(5, self, label="Script instance")

    def run_main_step(self):
        """Main processing step"""
        debug.trace(5, "Script.run_main_step(): self={self}")
        client = wikidata.client.Client()
        more_desired = True
        num_entities = 0
        num_random_entities = 0
        while ((num_entities < MAX_ENTITIES) and more_desired):
            num_random_entities += 1
            debug.trace(6, f"Getting next entity; good count: {num_entities}; (random count: {num_random_entities}")

            # Get random entity and its associated assertions
            all_assertions = []
            entity_num = 1 + random.randint(0, LAST_ID)
            entity_id = f"{ID_PREFIX}{entity_num}" 
            try:
                if (self.entity_hash and (not system.lookup_entry(self.entity_hash, entity_id))):
                    debug.trace(4, f"Excluding non-inclusion source entity {entity_id}")
                    continue
                num_entities += 1
                entity = client.get(entity_id)
                entity_name = str(entity.label)
                if (SIMPLE_NAMES and (not is_simple_name(entity_name))):
                    debug.trace(4, f"Excluding non-simple source {entity_id} ({entity_name})")
                    continue
                all_assertions = list(entity.items())
                debug.trace(6, f"entity: {entity}; assertions {all_assertions}")
                random.shuffle(all_assertions)
            except:
                debug.trace(4, f"Problem getting assertions for entity {entity_id}: {system.get_exception()}")
            # Show each assertion (up to max)
            num_assertions = 0
            for assertion in all_assertions[:MAX_PROPS]:
                debug.assertion(re.search("Entity P[0-9]+", str(assertion[0])))
                debug.trace(5, f"Resolving text for assertion: <{entity_id}, {assertion}>")

                # Print english-like representation
                try:
                    # Resolve relation and target names and see if assertion should be filtered
                    relation_name = str(assertion[0].label)
                    ## WTF?: debug.trace(5, "1")
                    target = assertion[1]
                    if not isinstance(target, wikidata.entity.Entity):
                        debug.trace(5, f"Ignoring non-entity target {target}")
                        continue
                    if target.type == wikidata.entity.EntityType.property:
                        debug.trace(5, f"Ignoring property {target}")
                        continue
                    target_id = target.id
                    debug.assertion(target_id.startswith(ID_PREFIX))
                    ## BAD: target_name = assertion[1] if isinstance(assertion[1], str) else str(assertion[1].label)
                    target_name = str(target.label)
                    ## WTF?: debug.trace(5, "2")
                    relationship = f"<{entity_name}, {relation_name}, {target_name}>"
                    ## WTF?: debug.trace(5, "4")
                    if (self.entity_hash and (not system.lookup_entry(self.entity_hash, target_id))):
                        ## WTF?: debug.trace(5, "4a")
                        debug.trace(4, f"Excluding non-inclusion target entity {target_id}")
                        ## WTF?: debug.trace(5, "4b")
                        continue
                    ## WTF?: debug.trace(5, "3")
                    if (SIMPLE_NAMES and (not is_simple_name(target_name))):
                        ## WTF?: debug.trace(5, "3a")
                        debug.trace(4, f"Excluding non-simple target {target_name}")
                        ## WTF?: debug.trace(5, "3b")
                        continue
                    ## WTF?: debug.trace(5, "5")
                    if (self.relation_hash and (not system.lookup_entry(self.relation_hash, relation_name))):
                        ## WTF?: debug.trace(5, "5a")
                        debug.trace(4, f"Excluding non-inclusion relation {relation_name}")
                        ## WTF?: debug.trace(5, "5b")
                        continue
                    ## WTF?: debug.trace(5, "6")
                    if (SKIP_REFLEXIVE and (entity_name.lower() == target_name.lower())):
                        ## WTF?: debug.trace(5, "6a")
                        debug.trace(4, f"Excluding reflexive relationship: {relationship}")
                        ## WTF?: debug.trace(5, "6n")
                        continue

                    ## WTF?: debug.trace(5, "7")
                    print(relationship)
                    num_assertions += 1
                    if (num_assertions == MAX_ASSERTIONS):
                        more_desired = False
                        break
                except:
                    debug.trace(4, f"Problem displaying {assertion}: {system.get_exception()}")
                    system.trace_stack(debug.VERBOSE)
                time.sleep(PAUSE_SECS)

#-------------------------------------------------------------------------------
    
if __name__ == '__main__':
    debug.trace_current_context(level=debug.QUITE_DETAILED)
    app = Script(
        description=__doc__,
        skip_input=False,
        manual_input=True,
        boolean_options=[TODO_ARG1],
        float_options=None)
    app.run()
