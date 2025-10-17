/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
#ifndef mach_o_Policy_h
#define mach_o_Policy_h

#include <string_view>

#include "MachODefines.h"
#include "Platform.h"
#include "Architecture.h"

namespace mach_o {

/*!
 * @class Policy
 *
 * @abstract
 *      Class for encapsulating policy for mach-o format details.
 *
 * @discussion
 *      The mach-o format is evolving over time. There are two categories
 *      of changes: new features and new restrictions.
 *
 *      A new feature is a new load command or new section, which only a new
 *      enough OS will understand. Each feature has a "use<xxx>()" method which
 *      ld checks to decide to emit a mach-o with the new feature.  The
 *      result of that method is a Usage value that specifies if the policy
 *      is to use or not use that feature, and if that use is a "must" or "preferred".
 *      A preferred policy can be overridden by a command line arg
 *      (e.g. -no\_fixup\_chains), whereas a must cannot be overridden.
 *
 *      A restriction is a constraint on existing mach-o details. These are driven
 *      by security, performance, or correctness concerns.  Each restriction
 *      has an "enforce<xxx>()" method which dyld and dyld\_info check to validate
 *      the binary. Restrictions are based on the SDK version the binary was built
 *      with. That is, the an old binary is allowed to violate the restriction,
 *      whereas a newer binary (build against newer SDK) is not.  The restriction
 *      logic is that the "enforce<xxx>()" will all return true for a binary built
 *      with the latest SDK.
 *
 */
class VIS_HIDDEN Policy
{
public:
                Policy(Architecture arch, PlatformAndVersions pvs, uint32_t filetype, bool pathMayBeInSharedCache=false, bool kernel=false, bool staticExec=false);

    enum Usage { preferUse, mustUse, preferDontUse, mustNotUse };

    // features
    Usage       useBuildVersionLoadCommand() const;
    Usage       useDataConst() const;
    Usage       useConstClassRefs() const;
    Usage       useGOTforClassRefs() const;
    Usage       useConstInterpose() const;
    Usage       useChainedFixups() const;
    Usage       useOpcodeFixups() const;
    Usage       useRelativeMethodLists() const;
    Usage       optimizeClassPatching() const;
    Usage       optimizeSingletonPatching() const;
    Usage       useAuthStubsInKexts() const;
    Usage       useDataConstForSelRefs() const;
    Usage       useSourceVersionLoadCommand() const;
    Usage       useLegacyLinkedit() const;
    bool        use4KBLoadCommandsPadding() const;
    bool        canUseDelayInit() const;
    uint16_t    chainedFixupsFormat() const;
    bool        useProtectedStack() const;

    // restrictions
    bool        enforceReadOnlyLinkedit() const;
    bool        enforceLinkeditContentAlignment() const;
    bool        enforceOneFixupEncoding() const;
    bool        enforceSegmentOrderMatchesLoadCmds() const;
    bool        enforceTextSegmentPermissions() const;
    bool        enforceFixupsInWritableSegments() const;
    bool        enforceCodeSignatureAligned() const;
    bool        enforceSectionsInSegment() const;
    bool        enforceHasLinkedDylibs() const;
    bool        enforceInstallNamesAreRealPaths() const;
    bool        enforceHasUUID() const;
    bool        enforceMainFlagsCorrect() const;
    bool        enforceNoDuplicateDylibs() const;
    bool        enforceNoDuplicateRPaths() const;
    bool        enforceDataSegmentPermissions() const;
    bool        enforceDataConstSegmentPermissions() const;
    bool        enforceImageListRemoveMainExecutable() const;
    bool        enforceSetSimulatorSharedCachePath() const;

private:
    bool              dyldLoadsOutput() const;
    bool              kernelOrKext() const;

    Platform::Epoch   _featureEpoch;
    Platform::Epoch   _enforcementEpoch;
    Architecture      _arch;
    PlatformAndVersions   _pvs;
    uint32_t          _filetype;
    bool              _pathMayBeInSharedCache;
    bool              _kernel;
    bool              _staticExec;
};




} // namespace mach_o

#endif // mach_o_Policy_h


