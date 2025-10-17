/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#ifndef BC_OPT_H
#define BC_OPT_H

#include <stdbool.h>
#include <stdlib.h>

/// The data required to parse command-line arguments.
typedef struct BcOpt
{
	/// The array of arguments.
	char** argv;

	/// The index of the current argument.
	size_t optind;

	/// The actual parse option character.
	int optopt;

	/// Where in the option we are for multi-character single-character options.
	int subopt;

	/// The option argument.
	char* optarg;

} BcOpt;

/// The types of arguments. This is specially adapted for bc.
typedef enum BcOptType
{
	/// No argument required.
	BC_OPT_NONE,

	/// An argument required.
	BC_OPT_REQUIRED,

	/// An option that is bc-only.
	BC_OPT_BC_ONLY,

	/// An option that is bc-only that requires an argument.
	BC_OPT_REQUIRED_BC_ONLY,

	/// An option that is dc-only.
	BC_OPT_DC_ONLY,

} BcOptType;

/// A struct to hold const data for long options.
typedef struct BcOptLong
{
	/// The name of the option.
	const char* name;

	/// The type of the option.
	BcOptType type;

	/// The character to return if the long option was parsed.
	int val;

} BcOptLong;

/**
 * Initialize data for parsing options.
 * @param o     The option data to initialize.
 * @param argv  The array of arguments.
 */
void
bc_opt_init(BcOpt* o, char** argv);

/**
 * Parse an option. This returns a value the same way getopt() and getopt_long()
 * do, so it returns a character for the parsed option or -1 if done.
 * @param o         The option data.
 * @param longopts  The long options.
 * @return          A character for the parsed option, or -1 if done.
 */
int
bc_opt_parse(BcOpt* o, const BcOptLong* longopts);

/**
 * Returns true if the option is `--` and not a long option.
 * @param a  The argument to parse.
 * @return   True if @a a is the `--` option, false otherwise.
 */
#define BC_OPT_ISDASHDASH(a) \
	((a) != NULL && (a)[0] == '-' && (a)[1] == '-' && (a)[2] == '\0')

/**
 * Returns true if the option is a short option.
 * @param a  The argument to parse.
 * @return   True if @a a is a short option, false otherwise.
 */
#define BC_OPT_ISSHORTOPT(a) \
	((a) != NULL && (a)[0] == '-' && (a)[1] != '-' && (a)[1] != '\0')

/**
 * Returns true if the option has `--` at the beginning, i.e., is a long option.
 * @param a  The argument to parse.
 * @return   True if @a a is a long option, false otherwise.
 */
#define BC_OPT_ISLONGOPT(a) \
	((a) != NULL && (a)[0] == '-' && (a)[1] == '-' && (a)[2] != '\0')

#endif // BC_OPT_H
