/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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
#ifndef _KERNELMANAGEMENT_SHIMS_H_
#define _KERNELMANAGEMENT_SHIMS_H_

bool shimmingEnabled();
CFStringRef kernelmanagement_path_for_bundle_id(CFStringRef);
OSReturn kernelmanagement_load_kext_url(CFURLRef, CFArrayRef);
OSReturn kernelmanagement_load_kext_identifier(CFStringRef, CFArrayRef);
OSReturn kernelmanagement_unload_kext_identifier(CFStringRef);

#endif /* _KERNELMANAGEMENT_SHIMS_H_ */
