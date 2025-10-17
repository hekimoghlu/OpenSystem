/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
#ifndef _CUPS_PWG_PRIVATE_H_
#  define _CUPS_PWG_PRIVATE_H_


/*
 * Include necessary headers...
 */

#  include <cups/cups.h>


/*
 * C++ magic...
 */

#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */


/*
 * Functions...
 */

extern void		_pwgGenerateSize(char *keyword, size_t keysize,
				         const char *prefix,
					 const char *name,
					 int width, int length)
					 _CUPS_INTERNAL_MSG("Use pwgFormatSizeName instead.");
extern int		_pwgInitSize(pwg_size_t *size, ipp_t *job,
				     int *margins_set)
				     _CUPS_INTERNAL_MSG("Use pwgInitSize instead.");
extern const pwg_media_t *_pwgMediaTable(size_t *num_media) _CUPS_PRIVATE;
extern pwg_media_t *_pwgMediaNearSize(int width, int length, int epsilon) _CUPS_PRIVATE;

#  ifdef __cplusplus
}
#  endif /* __cplusplus */

#endif /* !_CUPS_PWG_PRIVATE_H_ */
