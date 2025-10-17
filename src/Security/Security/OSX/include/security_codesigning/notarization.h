/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
#ifndef _H_NOTARIZATION
#define _H_NOTARIZATION

#include <Security/Security.h>
#include <security_utilities/dispatch.h>
#include <security_utilities/hashing.h>
#include <security_utilities/unix++.h>
#include "requirement.h"

namespace Security {
namespace CodeSigning {

// Performs an online check for a ticket, and returns true if a revocation ticket is found.
bool checkNotarizationServiceForRevocation(CFDataRef hash, SecCSDigestAlgorithm hashType, double *date);

// Performs an offline notarization check for the hash represented in the requirement context
// and returns whether the hash has a valid, unrevoked notarization ticket.
bool isNotarized(const Requirement::Context *context);

// Representation-specific methods for extracting a stapled ticket and registering
// it with the notarization daemon.
CFDataRef copyStapledTicketInBundle(const std::string& path);
void registerStapledTicketInPackage(const std::string& path);
void registerStapledTicketWithSystem(CFDataRef ticketData);

} // end namespace CodeSigning
} // end namespace Security

#endif /* _H_NOTARIZATION */
