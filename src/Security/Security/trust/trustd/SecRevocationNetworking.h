/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
#ifndef _SECURITY_SECREVOCATIONNETWORKING_H_
#define _SECURITY_SECREVOCATIONNETWORKING_H_

#import <CoreFoundation/CoreFoundation.h>
#import "trust/trustd/SecRevocationServer.h"

bool SecValidUpdateRequest(dispatch_queue_t queue, CFStringRef server, CFIndex version);
bool SecValidUpdateUpdateNow(dispatch_queue_t queue, CFStringRef server, CFIndex version);
bool SecORVCBeginFetches(SecORVCRef orvc, SecCertificateRef cert);

#endif /* _SECURITY_SECREVOCATIONNETWORKING_H_ */
