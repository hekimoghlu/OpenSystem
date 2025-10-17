/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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
#ifndef PremappedLoader_h
#define PremappedLoader_h

#include <TargetConditionals.h>

#include "JustInTimeLoader.h"
#include "Loader.h"

//
// PremappedLoaders:
//
// Premapped loaders are used in systems that have no disk.
// Binaries are mapped in memory ahead of time by the kernel.
// The images are then passed to dyld, which is in charge of applying fixups
// and performing any neccesaary initializing logic.

namespace dyld4 {

#if SUPPORT_CREATING_PREMAPPEDLOADERS
class PremappedLoader : public JustInTimeLoader
{
public:
    // these are the "virtual" methods that override Loader
    bool                        dyldDoesObjCFixups() const;
    void                        withLayout(Diagnostics &diag, const RuntimeState& state, void (^callback)(const mach_o::Layout &layout)) const;
    bool                        hasBeenFixedUp(RuntimeState&) const;
    bool                        beginInitializers(RuntimeState&);


    static Loader*      makePremappedLoader(Diagnostics& diag, RuntimeState& state, const char* path, bool isInDyldCache, uint32_t dylibCacheIndex, const LoadOptions& options, const mach_o::Layout* layout);
    static Loader*      makeLaunchLoader(Diagnostics& diag, RuntimeState& state, const MachOAnalyzer* mainExec, const char* mainExecPath, const mach_o::Layout* layout);

private:
    PremappedLoader(const MachOFile* mh, const Loader::InitialOptions& options, const mach_o::Layout* layout);
    static PremappedLoader*     make(RuntimeState& state, const MachOFile* mh, const char* path, bool willNeverUnload, bool overridesCache, uint16_t overridesDylibIndex, const mach_o::Layout* layout);


};
#endif // SUPPORT_CREATING_PREMAPPEDLOADERS

}  // namespace dyld4

#endif // PremappedLoader_h





