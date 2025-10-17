/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
/*!
 @header nameconstraints
 The functions provided in nameconstraints.h provide an interface to
 a name constraints implementation as specified in section 4.2.1.10 of rfc5280.
 */

#ifndef _SECURITY_NAMECONSTRAINTS_H_
#define _SECURITY_NAMECONSTRAINTS_H_

#include <stdbool.h>
#include <Security/SecCertificate.h>
#include <CoreFoundation/CFArray.h>

OSStatus SecNameContraintsMatchSubtrees(SecCertificateRef certificate, CFArrayRef subtrees, bool *matched, bool permit);

// Returns false if we encountered a subtree with an unsupported GN type
bool SecNameConstraintsAreSubtreesSupported(CFArrayRef subtrees);

void SecNameConstraintsIntersectSubtrees(CFMutableArrayRef subtrees_state, CFArrayRef subtrees_new);

#endif /* SECURITY_NAMECONSTRAINTS_H */
