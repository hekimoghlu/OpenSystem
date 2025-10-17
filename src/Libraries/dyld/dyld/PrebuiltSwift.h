/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
#ifndef PrebuiltSwift_h
#define PrebuiltSwift_h

#include "DyldRuntimeState.h"
#include "PrebuiltObjC.h"
#include "OptimizerSwift.h"
#include "Map.h"


namespace dyld4 {

#if SUPPORT_PREBUILTLOADERS || BUILDING_UNIT_TESTS || BUILDING_CACHE_BUILDER_UNIT_TESTS
//
// PrebuiltSwift computes read-only optimized data structures to store in the PrebuiltLoaderSet
//
struct PrebuiltSwift {

public:


    PrebuiltSwift() = default;
    ~PrebuiltSwift() = default;

    void make(Diagnostics& diag, PrebuiltObjC& prebuiltObjC, RuntimeState& state);

    TypeProtocolMap     typeProtocolConformances = { nullptr };
    MetadataProtocolMap metadataProtocolConformances = { nullptr };
    ForeignProtocolMap  foreignProtocolConformances = { nullptr };

    bool builtSwift              = false;

private:
    bool findProtocolConformances(Diagnostics& diag, PrebuiltObjC& prebuiltObjC, RuntimeState& state);

};
#endif // SUPPORT_PREBUILTLOADERS || BUILDING_UNIT_TESTS || BUILDING_CACHE_BUILDER_UNIT_TESTS
} // namespace dyld4



#endif /* PrebuiltSwift_h */
