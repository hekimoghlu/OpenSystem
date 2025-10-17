/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#import <Foundation/Foundation.h>

/**
 * Different definitions for the gcore generation.
 * There is a correspondence between the arguments to the framework and the command line:
 * \code
 *    SPI ARGUMENT               CLI Counterpart
 * --------------------------  ----------------------
 *   GCORE_OPTION_CORPSIFY        -C                                    Create a corpse for core file generation
 *   GCORE_OPTION_VERBOSE         -v                                    Add log information to stdout
 *   GCORE_OPTION_DEBUG           -d                                    Add debug information to stdout (cannot use more than once on framework), have a parameter (integer) with the debug level
 *   GCORE_OPTION_ANNOTATIONS     -N
 *   GCORE_OPTION_OUT_FILENAME    -o
 *   GCORE_OPTION_PID             ""                                    PID of process to create a gcore
 *   GCORE_OPTION_FD              "-f"                                  Use a file rather than a filename handle to perform file IO operations
 *\endcode
 * 
 */
#define GCORE_OPTION_CORPSIFY       "corpsify"
#define GCORE_OPTION_SUSPEND        "suspend"
#define GCORE_OPTION_VERBOSE        "verbose"
#ifdef CONFIG_DEBUG
#define GCORE_OPTION_DEBUG          "debug"
#endif
#define GCORE_OPTION_ANNOTATIONS    "annotations"
#define GCORE_OPTION_TASK_PORT      "port"
#define GCORE_OPTION_OUT_FILENAME   "filename"
#define GCORE_OPTION_PID            "pid"
#define GCORE_OPTION_FD             "filedesc"
/**
 * Create a coredump with the selected options, return 0 if the coredump was properly created or an error code.
 *
 * Possible errors are:
 *   EINVAL: One argument or a key is not a NSString, cannot process the value.
 *   EDOM:   One option is not recognized.
 *   ERANGE: One argument should have a parameter but is not found, or the data type is not a NSNumber
 *   ENOMEM: There was not enought memory for internal allocations
 *
 */
int create_gcore_with_options(NSDictionary *options);

