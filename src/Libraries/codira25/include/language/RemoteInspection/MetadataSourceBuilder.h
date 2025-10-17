/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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

//===--- MetadataSourceBuilder.h - Metadata Source Builder ------*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
// Implements utilities for constructing MetadataSources.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_REFLECTION_METADATASOURCEBUILDER_H
#define LANGUAGE_REFLECTION_METADATASOURCEBUILDER_H

#include "language/RemoteInspection/MetadataSource.h"

namespace language {
namespace reflection {

class MetadataSourceBuilder {
  std::vector<std::unique_ptr<const MetadataSource>> MetadataSourcePool;
public:
  using Source = const MetadataSource *;

  MetadataSourceBuilder() {}

  MetadataSourceBuilder(const MetadataSourceBuilder &Other) = delete;
  MetadataSourceBuilder &operator=(const MetadataSourceBuilder &Other) = delete;

  template <typename MetadataSourceTy, typename... Args>
  MetadataSourceTy *make_source(Args... args) {
    auto MS = new MetadataSourceTy(::std::forward<Args>(args)...);
    MetadataSourcePool.push_back(std::unique_ptr<const MetadataSource>(MS));
    return MS;
  }

  const GenericArgumentMetadataSource *
  createGenericArgument(unsigned Index, const MetadataSource *Source) {
    return GenericArgumentMetadataSource::create(*this, Index, Source);
  }

  const MetadataCaptureMetadataSource *
  createMetadataCapture(unsigned Index) {
    return MetadataCaptureMetadataSource::create(*this, Index);
  }

  const ReferenceCaptureMetadataSource *
  createReferenceCapture(unsigned Index) {
    return ReferenceCaptureMetadataSource::create(*this, Index);
  }

  const ClosureBindingMetadataSource *
  createClosureBinding(unsigned Index) {
    return ClosureBindingMetadataSource::create(*this, Index);
  }

  const SelfMetadataSource *
  createSelf() {
    return SelfMetadataSource::create(*this);
  }

  const SelfWitnessTableMetadataSource *
  createSelfWitnessTable() {
    return SelfWitnessTableMetadataSource::create(*this);
  }
};

} // end namespace reflection
} // end namespace language

#endif // LANGUAGE_REFLECTION_METADATASOURCEBUILDER_H
