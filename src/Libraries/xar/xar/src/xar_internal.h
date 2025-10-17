/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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

//
//  xar_internal.h
//  xarLib
//
//  Created by Jared Jones on 10/12/21.
//

#ifndef _XAR_INTERNAL_H_
#define _XAR_INTERNAL_H_

#ifdef XARSIG_BUILDING_WITH_XAR
#include "xar.h"
#else
#include <xar/xar.h>
#endif // XARSIG_BUILDING_WITH_XAR

// Undeprecate these for internal usage
xar_t xar_open(const char *file, int32_t flags) API_AVAILABLE(macos(10.4));
xar_t xar_open_digest_verify(const char *file, int32_t flags, void *expected_toc_digest, size_t expected_toc_digest_len) API_AVAILABLE(macos(10.14.4));
xar_t xar_fdopen_digest_verify(int fd, int32_t flags, void *expected_toc_digest, size_t expected_toc_digest_len) API_AVAILABLE(macos(15.0));
char *xar_get_path(xar_file_t f) API_AVAILABLE(macos(10.4));

#endif /* _XAR_INTERNAL_H_ */
