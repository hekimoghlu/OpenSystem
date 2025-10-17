/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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
//  utils.h
//  dcerpctest_server
//
//  Created by William Conway on 12/1/23.
//

#ifndef utils_h
#define utils_h

// Prints a description of a given error_status_t code.
// Arguments
//     ecode:   an rpc error_status_t code.
//     routine: Otional routine name which returned the error.
//     ctx:    Optional context about the error.
//     fatal:   Calls exit(1) if not set to zero.
//
void  chk_dce_err(error_status_t, const char *, const char *, unsigned int);

#endif /* utils_h */
