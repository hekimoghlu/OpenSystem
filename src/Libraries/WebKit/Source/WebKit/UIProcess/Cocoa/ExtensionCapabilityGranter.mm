/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
#import "ExtensionCapabilityGranter.h"

#if ENABLE(EXTENSION_CAPABILITIES)

#import "ExtensionCapability.h"
#import "ExtensionCapabilityGrant.h"
#import "ExtensionKitSPI.h"
#import "GPUProcessProxy.h"
#import "MediaCapability.h"
#import "WebProcessProxy.h"
#import <BrowserEngineKit/BrowserEngineKit.h>
#import <wtf/CrossThreadCopier.h>
#import <wtf/NativePromise.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/TZoneMallocInlines.h>

#define GRANTER_RELEASE_LOG(envID, fmt, ...) RELEASE_LOG(ProcessCapabilities, "%{public}s[envID=%{public}s] " fmt, __FUNCTION__, envID.utf8().data(), ##__VA_ARGS__)
#define GRANTER_RELEASE_LOG_ERROR(envID, fmt, ...) RELEASE_LOG_ERROR(ProcessCapabilities, "%{public}s[envID=%{public}s] " fmt, __FUNCTION__, envID.utf8().data(), ##__VA_ARGS__)

namespace WebKit {

static WorkQueue& granterQueue()
{
    static NeverDestroyed<Ref<WorkQueue>> queue(WorkQueue::create("ExtensionCapabilityGranter Queue"_s, WorkQueue::QOS::UserInitiated));
    return queue.get();
}

#if USE(EXTENSIONKIT)
static PlatformGrant grantCapability(const PlatformCapability& capability, const std::optional<ExtensionProcess>& process)
{
    if (!ExtensionCapability::platformCapabilityIsValid(capability) || !process)
        return { };

    return process->grantCapability(capability);
}
#endif

struct PlatformExtensionCapabilityGrants {
    PlatformGrant gpuProcessGrant;
    PlatformGrant webProcessGrant;

    PlatformExtensionCapabilityGrants isolatedCopy() const & { return { gpuProcessGrant, webProcessGrant }; }
};

enum class ExtensionCapabilityGrantError: uint8_t {
    PlatformError,
};

using ExtensionCapabilityGrantsPromise = NativePromise<PlatformExtensionCapabilityGrants, ExtensionCapabilityGrantError>;

static Ref<ExtensionCapabilityGrantsPromise> grantCapabilityInternal(const ExtensionCapability& capability, const GPUProcessProxy* gpuProcess, const WebProcessProxy* webProcess)
{
    auto gpuExtension = gpuProcess ? gpuProcess->extensionProcess() : std::nullopt;
    auto webExtension = webProcess ? webProcess->extensionProcess() : std::nullopt;

#if USE(EXTENSIONKIT)
    return invokeAsync(granterQueue(), [
        capability = capability.platformCapability(),
        gpuExtension = WTFMove(gpuExtension),
        webExtension = WTFMove(webExtension)
    ] {
        PlatformExtensionCapabilityGrants grants {
            grantCapability(capability, gpuExtension),
            grantCapability(capability, webExtension)
        };
        return ExtensionCapabilityGrantsPromise::createAndResolve(WTFMove(grants));
    });
#else
    UNUSED_PARAM(capability);
    return ExtensionCapabilityGrantsPromise::createAndReject(ExtensionCapabilityGrantError::PlatformError);
#endif
}

static bool prepareGrant(const String& environmentIdentifier, AuxiliaryProcessProxy& auxiliaryProcess)
{
    ExtensionCapabilityGrant grant { environmentIdentifier };
    auto& existingGrants = auxiliaryProcess.extensionCapabilityGrants();

    auto result = existingGrants.add(environmentIdentifier, WTFMove(grant));
    if (result.isNewEntry)
        return true;

    auto& existingGrant = result.iterator->value;
    if (existingGrant.isEmpty() || existingGrant.isValid())
        return false;

    existingGrants.set(environmentIdentifier, WTFMove(grant));
    return true;
}

static bool finalizeGrant(ExtensionCapabilityGranter& granter, const String& environmentIdentifier, AuxiliaryProcessProxy* auxiliaryProcess, ExtensionCapabilityGrant&& grant)
{
    if (!auxiliaryProcess) {
        GRANTER_RELEASE_LOG_ERROR(environmentIdentifier, "auxiliaryProcess is null");
        return false;
    }

    auto& existingGrants = auxiliaryProcess->extensionCapabilityGrants();

    auto iterator = existingGrants.find(environmentIdentifier);
    if (iterator == existingGrants.end()) {
        GRANTER_RELEASE_LOG_ERROR(environmentIdentifier, "grant previously revoked");
        return false;
    }

    if (grant.isValid()) {
        auto& existingGrant = iterator->value;
        ASSERT(!existingGrant.isValid());
        if (existingGrant.isValid()) {
            GRANTER_RELEASE_LOG_ERROR(environmentIdentifier, "grant not expected to be valid");
            granter.invalidateGrants(Vector<ExtensionCapabilityGrant>::from(WTFMove(existingGrant)));
        }
        existingGrant = WTFMove(grant);
        return true;
    }

    GRANTER_RELEASE_LOG_ERROR(environmentIdentifier, "grant invalid");
    existingGrants.remove(iterator);
    return false;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(ExtensionCapabilityGranter);

RefPtr<ExtensionCapabilityGranter> ExtensionCapabilityGranter::create(ExtensionCapabilityGranterClient& client)
{
    return adoptRef(new ExtensionCapabilityGranter(client));
}

ExtensionCapabilityGranter::ExtensionCapabilityGranter(ExtensionCapabilityGranterClient& client)
    : m_client { client }
{
}

void ExtensionCapabilityGranter::grant(const ExtensionCapability& capability)
{
    String environmentIdentifier = capability.environmentIdentifier();
    if (environmentIdentifier.isEmpty()) {
        GRANTER_RELEASE_LOG_ERROR(environmentIdentifier, "environmentIdentifier must not be empty");
        return;
    }

    RefPtr gpuProcess = m_client->gpuProcessForCapabilityGranter(*this);
    bool needsGPUProcessGrant = gpuProcess && prepareGrant(environmentIdentifier, *gpuProcess);

    RefPtr webProcess = m_client->webProcessForCapabilityGranter(*this, environmentIdentifier);
    bool needsWebProcessGrant = webProcess && prepareGrant(environmentIdentifier, *webProcess);

    if (!needsGPUProcessGrant && !needsWebProcessGrant)
        return;

    grantCapabilityInternal(
        capability,
        needsGPUProcessGrant ? gpuProcess.get() : nullptr,
        needsWebProcessGrant ? webProcess.get() : nullptr
    )->whenSettled(RunLoop::main(), [
        this,
        weakThis = WeakPtr { *this },
        environmentIdentifier,
        needsGPUProcessGrant,
        needsWebProcessGrant
    ](auto&& result) {
        ExtensionCapabilityGrant gpuProcessGrant { environmentIdentifier };
        ExtensionCapabilityGrant webProcessGrant { environmentIdentifier };
        if (result) {
            gpuProcessGrant.setPlatformGrant(WTFMove(result->gpuProcessGrant));
            webProcessGrant.setPlatformGrant(WTFMove(result->webProcessGrant));
        }
        RefPtr protectedThis = weakThis.get();
        RefPtr client = protectedThis ? m_client.ptr() : nullptr;
        if (!client) {
            invalidateGrants(Vector<ExtensionCapabilityGrant>::from(WTFMove(gpuProcessGrant), WTFMove(webProcessGrant)));
            return;
        }

        Vector<ExtensionCapabilityGrant> grantsToInvalidate;
        grantsToInvalidate.reserveInitialCapacity(2);

        if (needsGPUProcessGrant) {
            if (finalizeGrant(*this, environmentIdentifier, m_client->gpuProcessForCapabilityGranter(*this).get(), WTFMove(gpuProcessGrant)))
                GRANTER_RELEASE_LOG(environmentIdentifier, "granted for GPU process");
            else {
                GRANTER_RELEASE_LOG_ERROR(environmentIdentifier, "failed to grant for GPU process");
                grantsToInvalidate.append(WTFMove(gpuProcessGrant));
            }
        } else
            ASSERT(gpuProcessGrant.isEmpty());

        if (needsWebProcessGrant) {
            if (finalizeGrant(*this, environmentIdentifier, m_client->webProcessForCapabilityGranter(*this, environmentIdentifier).get(), WTFMove(webProcessGrant)))
                GRANTER_RELEASE_LOG(environmentIdentifier, "granted for WebContent process");
            else {
                GRANTER_RELEASE_LOG_ERROR(environmentIdentifier, "failed to grant for WebContent process");
                grantsToInvalidate.append(WTFMove(webProcessGrant));
            }
        } else
            ASSERT(webProcessGrant.isEmpty());

        invalidateGrants(WTFMove(grantsToInvalidate));
    });
}

void ExtensionCapabilityGranter::revoke(const ExtensionCapability& capability)
{
    Vector<ExtensionCapabilityGrant> grants;
    grants.reserveInitialCapacity(2);

    String environmentIdentifier = capability.environmentIdentifier();

    if (RefPtr client = m_client.ptr()) {
        if (RefPtr gpuProcess = m_client->gpuProcessForCapabilityGranter(*this))
            grants.append(gpuProcess->extensionCapabilityGrants().take(environmentIdentifier));

        if (RefPtr webProcess = m_client->webProcessForCapabilityGranter(*this, environmentIdentifier))
            grants.append(webProcess->extensionCapabilityGrants().take(environmentIdentifier));
    }

    invalidateGrants(WTFMove(grants));
}

using ExtensionCapabilityActivationPromise = NativePromise<void, ExtensionCapabilityGrantError>;

void ExtensionCapabilityGranter::setMediaCapabilityActive(MediaCapability& capability, bool isActive)
{
    switch (capability.state()) {
    case MediaCapability::State::Inactive:
    case MediaCapability::State::Deactivating:
        if (!isActive)
            return;
        capability.setState(MediaCapability::State::Activating);
        break;
    case MediaCapability::State::Activating:
    case MediaCapability::State::Active:
        if (isActive)
            return;
        capability.setState(MediaCapability::State::Deactivating);
        break;
    }

    GRANTER_RELEASE_LOG(capability.environmentIdentifier(), "%{public}s", isActive ? "activating" : "deactivating");

    invokeAsync(granterQueue(), [platformCapability = capability.platformCapability(), platformMediaEnvironment = RetainPtr { capability.platformMediaEnvironment() }, isActive] {
#if USE(EXTENSIONKIT)
        NSError *error = nil;
        if (isActive)
            [platformMediaEnvironment activateWithError:&error];
        else
            [platformMediaEnvironment suspendWithError:&error];
        if (error)
            RELEASE_LOG_ERROR(ProcessCapabilities, "%{public}s failed with error: %{public}@", __FUNCTION__, error);
        else
            return ExtensionCapabilityActivationPromise::createAndResolve();
#endif
        return ExtensionCapabilityActivationPromise::createAndReject(ExtensionCapabilityGrantError::PlatformError);
    })->whenSettled(RunLoop::main(), [weakCapability = WeakPtr { capability }, isActive](auto&& result) {
        auto capability = weakCapability.get();
        if (!capability)
            return;

        if (!result) {
            GRANTER_RELEASE_LOG_ERROR(capability->environmentIdentifier(), "failed to %{public}s", isActive ? "activate" : "deactivate");
            capability->setState(MediaCapability::State::Inactive);
            return;
        }

        switch (capability->state()) {
        case MediaCapability::State::Deactivating:
            if (!isActive)
                capability->setState(MediaCapability::State::Inactive);
            break;
        case MediaCapability::State::Activating:
            if (isActive)
                capability->setState(MediaCapability::State::Active);
            break;
        case MediaCapability::State::Inactive:
        case MediaCapability::State::Active:
            ASSERT_NOT_REACHED();
            return;
        }

        GRANTER_RELEASE_LOG(capability->environmentIdentifier(), "%{public}s", isActive ? "activated" : "deactivated");
    });
}

void ExtensionCapabilityGranter::invalidateGrants(Vector<ExtensionCapabilityGrant>&& grants)
{
    granterQueue().dispatch([grants = crossThreadCopy(WTFMove(grants))]() mutable {
        for (auto& grant : grants)
            grant.setPlatformGrant({ });
    });
}

} // namespace WebKit

#undef GRANTER_RELEASE_LOG
#undef GRANTER_RELEASE_LOG_ERROR

#endif // ENABLE(EXTENSION_CAPABILITIES)
