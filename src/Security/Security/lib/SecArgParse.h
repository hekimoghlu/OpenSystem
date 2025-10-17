/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef SecArgParse_h
#define SecArgParse_h

#include <getopt.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/* This is a poor simulacrum of python's argparse library. To use, create an arguments struct, including an array of
 * argument elements, and pass the struct (along with argv+argc) to options_parse(). This is one-shot argument parsing:
 * you must set pointers for *argument and *flag in each option to receive the results of the argument parsing.
 *
 * Currently does not support:
 *  non-string arguments
 *  default values
 *  relationships between arguments
 *  detecting meaningless option configurations
 *
 * Example arguments:
 *     static struct argument options[] = {
 *       { .shortname='p', .longname="perfcounters", .flag=&perfCounters, .flagval=true, .description="Print performance counters"},
 *       { .longname="test", .flag=&test, .flagval=true, .description="test long option"},
 *       { .command="resync", .flag=&resync, .flagval=true, .description="Initiate a resync"},
 *       { .positional_name="positional", .positional_optional=false, .argument=&position, .description = "Positional argument" },
 *       { .shortname='a', .longname="asdf", .argname="number", .argument=&asdf, .description = "Long arg with argument" },
 *     };
 *
 *     static struct arguments args = {
 *       .programname="testctl",
 *       .description="Control and report",
 *       .arguments = options,
 *     };
 *
 *  Note: this library automatically adds a '-h' and a '--help' operation. Don't try to override this.
 */

struct argument {
    char   shortname;
    char*  longname;
    char*  command;
    char*  positional_name;
    bool   positional_optional;
    char*  argname;

    char** argument;
    int*   flag;
    int    flagval;
    char*  description;
    bool   internal_only;

    char *** argument_array;
    size_t* argument_array_count;
};

struct arguments {
    char* programname;
    char* description;

    struct argument* arguments;
};

bool options_parse(int argc, char * const *argv, const struct arguments* args);
void print_usage(const struct arguments* args);

#endif /* SecArgParse_h */
