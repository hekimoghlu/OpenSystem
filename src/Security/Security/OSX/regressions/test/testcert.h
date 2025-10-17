/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
#include <Security/SecIdentity.h>
#include <Security/SecCertificate.h>
#include <Security/SecKey.h>

OSStatus
test_cert_generate_key(uint32_t key_size_in_bits, CFTypeRef sec_attr_key_type,
                       SecKeyRef *private_key, SecKeyRef *public_key);

SecIdentityRef
test_cert_create_root_certificate(CFStringRef subject,
	SecKeyRef public_key, SecKeyRef private_key);

CF_RETURNS_RETAINED
SecCertificateRef
test_cert_issue_certificate(SecIdentityRef ca_identity,
	SecKeyRef public_key, CFStringRef subject,
	unsigned int serial_no, unsigned int key_usage);

CF_RETURNS_RETAINED
CFArrayRef test_cert_string_to_subject(CFStringRef subject);
