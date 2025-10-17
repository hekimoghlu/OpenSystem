/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
#include "config.h"
#include "Permissions.h"

#include "DedicatedWorkerGlobalScope.h"
#include "Document.h"
#include "Exception.h"
#include "JSDOMPromiseDeferred.h"
#include "JSPermissionDescriptor.h"
#include "JSPermissionStatus.h"
#include "LocalFrame.h"
#include "NavigatorBase.h"
#include "Page.h"
#include "PermissionController.h"
#include "PermissionDescriptor.h"
#include "PermissionName.h"
#include "PermissionQuerySource.h"
#include "PermissionsPolicy.h"
#include "ScriptExecutionContext.h"
#include "SecurityOrigin.h"
#include "ServiceWorkerGlobalScope.h"
#include "SharedWorkerGlobalScope.h"
#include "WorkerGlobalScope.h"
#include "WorkerLoaderProxy.h"
#include "WorkerThread.h"
#include <optional>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/TypeCasts.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Permissions);

Ref<Permissions> Permissions::create(NavigatorBase& navigator)
{
    return adoptRef(*new Permissions(navigator));
}

Permissions::Permissions(NavigatorBase& navigator)
    : m_navigator(navigator)
{
}

NavigatorBase* Permissions::navigator()
{
    return m_navigator.get();
}

Permissions::~Permissions() = default;

static bool isAllowedByPermissionsPolicy(const Document& document, PermissionName name)
{
    switch (name) {
    case PermissionName::Camera:
        return PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Camera, document, PermissionsPolicy::ShouldReportViolation::No);
    case PermissionName::Geolocation:
        return PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Geolocation, document, PermissionsPolicy::ShouldReportViolation::No);
    case PermissionName::Microphone:
        return PermissionsPolicy::isFeatureEnabled(PermissionsPolicy::Feature::Microphone, document, PermissionsPolicy::ShouldReportViolation::No);
    default:
        return true;
    }
}

std::optional<PermissionQuerySource> Permissions::sourceFromContext(const ScriptExecutionContext& context)
{
    if (is<Document>(context))
        return PermissionQuerySource::Window;
    if (is<DedicatedWorkerGlobalScope>(context))
        return PermissionQuerySource::DedicatedWorker;
    if (is<SharedWorkerGlobalScope>(context))
        return PermissionQuerySource::SharedWorker;
    if (is<ServiceWorkerGlobalScope>(context))
        return PermissionQuerySource::ServiceWorker;
    return std::nullopt;
}


std::optional<PermissionName> Permissions::toPermissionName(const String& name)
{
    if (name == "camera"_s)
        return PermissionName::Camera;
    if (name == "geolocation"_s)
        return PermissionName::Geolocation;
    if (name == "microphone"_s)
        return PermissionName::Microphone;
    if (name == "notifications"_s)
        return PermissionName::Notifications;
    if (name == "push"_s)
        return PermissionName::Push;
    return std::nullopt;
}

void Permissions::query(JSC::Strong<JSC::JSObject> permissionDescriptorValue, DOMPromiseDeferred<IDLInterface<PermissionStatus>>&& promise)
{
    RefPtr context = m_navigator ? m_navigator->scriptExecutionContext() : nullptr;
    if (!context || !context->globalObject()) {
        promise.reject(Exception { ExceptionCode::InvalidStateError, "The context is invalid"_s });
        return;
    }

    auto source = sourceFromContext(*context);
    if (!source) {
        promise.reject(Exception { ExceptionCode::NotSupportedError, "Permissions::query is not supported in this context"_s  });
        return;
    }

    RefPtr document = dynamicDowncast<Document>(*context);
    if (document && !document->isFullyActive()) {
        promise.reject(Exception { ExceptionCode::InvalidStateError, "The document is not fully active"_s });
        return; 
    }

    auto& vm = context->globalObject()->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto permissionDescriptorConversionResult = convert<IDLDictionary<PermissionDescriptor>>(*context->globalObject(), permissionDescriptorValue.get());
    if (UNLIKELY(permissionDescriptorConversionResult.hasException(scope))) {
        promise.reject(Exception { ExceptionCode::ExistingExceptionError });
        return;
    }

    auto permissionDescriptor = permissionDescriptorConversionResult.releaseReturnValue();

    RefPtr origin = context->securityOrigin();
    auto originData = origin ? origin->data() : SecurityOriginData { };

    if (document) {
        WeakPtr page = document->page();
        if (!page) {
            promise.reject(Exception { ExceptionCode::InvalidStateError, "The page does not exist"_s });
            return;
        }

        if (!isAllowedByPermissionsPolicy(*document, permissionDescriptor.name)) {
            promise.resolve(PermissionStatus::create(*context, PermissionState::Denied, permissionDescriptor, PermissionQuerySource::Window, *page));
            return;
        }

        PermissionController::shared().query(ClientOrigin { document->topOrigin().data(), WTFMove(originData) }, permissionDescriptor, *page, *source, [document = Ref { *document }, page, permissionDescriptor, promise = WTFMove(promise)](auto permissionState) mutable {
            if (!permissionState) {
                promise.reject(Exception { ExceptionCode::NotSupportedError, "Permissions::query does not support this API"_s });
                return;
            }

            promise.resolve(PermissionStatus::create(document, *permissionState, permissionDescriptor, PermissionQuerySource::Window, WTFMove(page)));
        });
        return;
    }

    auto& workerGlobalScope = downcast<WorkerGlobalScope>(*context);
    auto completionHandler = [originData = WTFMove(originData).isolatedCopy(), permissionDescriptor, contextIdentifier = workerGlobalScope.identifier(), source = *source, promise = WTFMove(promise)] (auto& context) mutable {
        ASSERT(isMainThread());

        auto& document = downcast<Document>(context);
        if (!document.page()) {
            ScriptExecutionContext::postTaskTo(contextIdentifier, [promise = WTFMove(promise)](auto&) mutable {
                promise.reject(Exception { ExceptionCode::InvalidStateError, "The page does not exist"_s });
            });
            return;
        }

        auto page = source == PermissionQuerySource::DedicatedWorker ? WeakPtr { *document.page() } : nullptr;

        PermissionController::shared().query(ClientOrigin { document.topOrigin().data(), WTFMove(originData) }, permissionDescriptor, page, source, [contextIdentifier, permissionDescriptor, promise = WTFMove(promise), source, page](auto permissionState) mutable {
            ScriptExecutionContext::postTaskTo(contextIdentifier, [promise = WTFMove(promise), permissionState, permissionDescriptor, source, page = WTFMove(page)](auto& context) mutable {
                if (!permissionState) {
                    promise.reject(Exception { ExceptionCode::NotSupportedError, "Permissions::query does not support this API"_s });
                    return;
                }

                promise.resolve(PermissionStatus::create(context, *permissionState, permissionDescriptor, source, WTFMove(page)));
            });
        });
    };

    if (auto* workerLoaderProxy = workerGlobalScope.thread().workerLoaderProxy())
        workerLoaderProxy->postTaskToLoader(WTFMove(completionHandler));
}

} // namespace WebCore
