/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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
#import "config.h"
#import "AuxiliaryProcess.h"

#if PLATFORM(IOS_FAMILY) && !PLATFORM(MACCATALYST)

#import "SandboxInitializationParameters.h"
#import "XPCServiceEntryPoint.h"
#import <WebCore/FloatingPointEnvironment.h>
#import <WebCore/SystemVersion.h>
#import <mach/mach.h>
#import <mach/task.h>
#import <pal/spi/ios/MobileGestaltSPI.h>
#import <pwd.h>
#import <stdlib.h>
#import <sysexits.h>
#import <wtf/FileSystem.h>

namespace WebKit {

void AuxiliaryProcess::initializeSandbox(const AuxiliaryProcessInitializationParameters&, SandboxInitializationParameters&)
{
    RELEASE_ASSERT_NOT_REACHED();
}

void AuxiliaryProcess::setQOS(int, int)
{

}

void AuxiliaryProcess::populateMobileGestaltCache(std::optional<SandboxExtension::Handle>&& mobileGestaltExtensionHandle)
{
    if (!mobileGestaltExtensionHandle)
        return;

    if (auto extension = SandboxExtension::create(WTFMove(*mobileGestaltExtensionHandle))) {
        bool ok = extension->consume();
        ASSERT_UNUSED(ok, ok);
        // If we have an extension handle for MobileGestalt, it means the MobileGestalt cache is invalid.
        // In this case, we perform a set of MobileGestalt queries while having access to the daemon,
        // which will populate the MobileGestalt in-memory cache with correct values.
        // The set of queries below was determined by finding all possible queries that have cachable
        // values, and would reach out to the daemon for the answer. That way, the in-memory cache
        // should be identical to a valid MobileGestalt cache after having queried all of these values.
        MGGetFloat32Answer(kMGQMainScreenScale, 0);
        MGGetSInt32Answer(kMGQMainScreenPitch, 0);
        MGGetSInt32Answer(kMGQMainScreenClass, MGScreenClassPad2);
        MGGetBoolAnswer(kMGQAppleInternalInstallCapability);
        MGGetBoolAnswer(kMGQiPadCapability);
        auto deviceName = adoptCF(MGCopyAnswer(kMGQDeviceName, nullptr));
        MGGetSInt32Answer(kMGQDeviceClassNumber, MGDeviceClassInvalid);
        MGGetBoolAnswer(kMGQHasExtendedColorDisplay);
        MGGetFloat32Answer(kMGQDeviceCornerRadius, 0);
        MGGetBoolAnswer(kMGQSupportsForceTouch);

        auto answer = adoptCF(MGCopyAnswer(kMGQBluetoothCapability, nullptr));
        answer = MGCopyAnswer(kMGQDeviceProximityCapability, nullptr);
        answer = MGCopyAnswer(kMGQDeviceSupportsARKit, nullptr);
        answer = MGCopyAnswer(kMGQTimeSyncCapability, nullptr);
        answer = MGCopyAnswer(kMGQWAPICapability, nullptr);
        answer = MGCopyAnswer(kMGQMainDisplayRotation, nullptr);

        ok = extension->revoke();
        ASSERT_UNUSED(ok, ok);
    }
}

} // namespace WebKit

#endif
