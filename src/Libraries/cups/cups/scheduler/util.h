/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
#ifndef _CUPSD_UTIL_H_
#  define _CUPSD_UTIL_H_

/*
 * Include necessary headers...
 */

#  include <cups/array-private.h>
#  include <cups/file-private.h>
#  include <signal.h>


/*
 * C++ magic...
 */

#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */


/*
 * Types...
 */

typedef int (*cupsd_compare_func_t)(const void *, const void *);


/*
 * Prototypes...
 */

extern int		cupsdCompareNames(const char *s, const char *t);
extern cups_array_t	*cupsdCreateStringsArray(const char *s);
extern int		cupsdExec(const char *command, char **argv);
extern cups_file_t	*cupsdPipeCommand(int *pid, const char *command,
			                  char **argv, uid_t user);
extern void		cupsdSendIPPGroup(ipp_tag_t group_tag);
extern void		cupsdSendIPPHeader(ipp_status_t status_code,
			                   int request_id);
extern void		cupsdSendIPPInteger(ipp_tag_t value_tag,
			                    const char *name, int value);
extern void		cupsdSendIPPString(ipp_tag_t value_tag,
			                   const char *name, const char *value);
extern void		cupsdSendIPPTrailer(void);


#  ifdef __cplusplus
}
#  endif /* __cplusplus */

#endif /* !_CUPSD_UTIL_H_ */
