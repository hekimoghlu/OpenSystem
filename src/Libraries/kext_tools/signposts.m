/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
//  signposts.m
//  kext_tools
//
//  Copyright 2018 Apple Inc. All rights reserved.
//
#ifndef EMBEDDED_HOST
#import <Foundation/Foundation.h>

#import "kext_tools_util.h"
#import "signposts.h"

void
signpost_kext_properties(OSKextRef theKext, os_signpost_id_t spid)
{
    os_signpost_event_emit(get_signpost_log(), spid, "KextURL", "%@", OSKextGetURL(theKext));
    os_signpost_event_emit(get_signpost_log(), spid, "KextBundleID", "%@", OSKextGetIdentifier(theKext));
}

os_signpost_id_t
generate_signpost_id(void)
{
    return os_signpost_id_generate(get_signpost_log());
}

#endif /* !EMBEDDED_HOST: until the builders support 10.14, we need to disable signpost APIs in the embedded-host tools */
