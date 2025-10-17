/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#include <CoreFoundation/CoreFoundation.h>
#include <security_asn1/SecAsn1Coder.h>

#ifndef _SECURITY_P12IMPORT_H_
#define _SECURITY_P12IMPORT_H_

__BEGIN_DECLS

typedef enum {
    p12_noErr = 0,
    p12_decodeErr,
    p12_passwordErr,
} p12_error;

typedef struct {
	SecAsn1CoderRef coder;
    CFStringRef passphrase;
    CFMutableDictionaryRef items;
} pkcs12_context;

p12_error p12decode(pkcs12_context * context, CFDataRef cdpfx);

__END_DECLS

#endif /* !_SECURITY_P12IMPORT_H_ */
