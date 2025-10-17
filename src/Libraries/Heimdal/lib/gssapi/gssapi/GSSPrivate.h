/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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


#ifndef _GSS_FRAMEWORK_PRIVATE
#define _GSS_FRAMEWORK_PRIVATE 1

__nullable CFStringRef
GSSRuleGetMatch(__nonnull CFDictionaryRef rules, __nonnull CFStringRef hostname)
;

void
GSSRuleAddMatch(__nonnull CFMutableDictionaryRef rules, __nonnull CFStringRef host, __nonnull CFStringRef value);

/*
 * Cred
 */

gss_name_t
GSSCredCopyName(__nonnull gss_cred_id_t cred);

OM_uint32
GSSCredGetLifetime(__nonnull gss_cred_id_t cred);


#endif /* _GSS_FRAMEWORK_PRIVATE */
