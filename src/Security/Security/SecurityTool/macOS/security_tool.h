/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
#ifndef _SECURITY_TOOL_H_
#define _SECURITY_TOOL_H_ 1

#define SHOW_USAGE_MESSAGE 2

#ifdef __cplusplus
extern "C" {
#endif

typedef int(*command_func)(int argc, char * const *argv);

/* If 1 attempt to be as quiet as possible. */
extern int do_quiet;

/* If 1 attempt to be as verbose as possible. */
extern int do_verbose;

const char *sec_errstr(int err);
void sec_error(const char *msg, ...) __attribute((format(printf, 1, 2)));
void sec_perror(const char *msg, int err);

#ifdef __cplusplus
}
#endif

#endif /*  _SECURITY_TOOL_H_ */
