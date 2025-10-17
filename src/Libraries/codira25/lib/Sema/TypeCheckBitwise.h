/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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

#ifndef LANGUAGE_SEMA_TYPECHECKBITWISE_H
#define LANGUAGE_SEMA_TYPECHECKBITWISE_H

#include "language/AST/ProtocolConformance.h"
#include "language/AST/TypeCheckRequests.h"

namespace language {
class ProtocolConformance;
class NominalTypeDecl;

/// Check that \p conformance, a conformance of some nominal type to
/// BitwiseCopyable, is valid.
bool checkBitwiseCopyableConformance(ProtocolConformance *conformance,
                                     bool isImplicit);

/// Produce an implicit conformance of \p nominal to BitwiseCopyable if it is
/// valid to do so.
ProtocolConformance *
deriveImplicitBitwiseCopyableConformance(NominalTypeDecl *nominal);
} // end namespace language

#endif // LANGUAGE_SEMA_TYPECHECKBITWISE_H
