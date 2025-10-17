/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#ifndef _PPDX_H_
#  define _PPDX_H_


/*
 * Include necessary headers...
 */

#  include <cups/ppd.h>


/*
 * C++ magic...
 */

#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */


/*
 * Maximum amount of data to encode/decode...
 */

#  define PPDX_MAX_STATUS	1024	/* Limit on log messages in 10.6 */
#  define PPDX_MAX_DATA		16777216/* 16MiB */


/*
 * 'ppdxReadData()' - Read encoded data from a ppd_file_t *.
 *
 * Reads chunked data in the PPD file "ppd" using the prefix "name".  Returns
 * an allocated pointer to the data (which is nul-terminated for convenience)
 * along with the length of the data in the variable pointed to by "datasize",
 * which can be NULL to indicate the caller doesn't need the length.
 *
 * Returns NULL if no data is present in the PPD with the prefix.
 */

extern void	*ppdxReadData(ppd_file_t *ppd, const char *name,
		              size_t *datasize);


/*
 * 'ppdxWriteData()' - Writes encoded data to stderr using PPD: messages.
 *
 * Writes chunked data to the PPD file using PPD: messages sent to stderr for
 * cupsd.  "name" must be a valid PPD keyword string whose length is less than
 * 37 characters to allow for chunk numbering.  "data" provides a pointer to the
 * data to be written, and "datasize" provides the length.
 */

extern void	ppdxWriteData(const char *name, const void *data,
			      size_t datasize);


#  ifdef __cplusplus
}
#  endif /* __cplusplus */

#endif /* !_PPDX_H */
