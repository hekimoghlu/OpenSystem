/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
/* OS/400 additional definitions. */

#ifndef __OS400_SYS_
#define __OS400_SYS_


/* Per-thread item identifiers. */

typedef enum {
        LK_GSK_ERROR,
        LK_LDAP_ERROR,
        LK_CURL_VERSION,
        LK_VERSION_INFO,
        LK_VERSION_INFO_DATA,
        LK_EASY_STRERROR,
        LK_SHARE_STRERROR,
        LK_MULTI_STRERROR,
        LK_URL_STRERROR,
        LK_ZLIB_VERSION,
        LK_ZLIB_MSG,
        LK_LAST
}               localkey_t;


extern char *   (* Curl_thread_buffer)(localkey_t key, long size);


/* Maximum string expansion factor due to character code conversion. */

#define MAX_CONV_EXPANSION      4       /* Can deal with UTF-8. */

#endif
