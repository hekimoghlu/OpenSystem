/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#include "NavigatorEME.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "CDM.h"
#include "CDMLogging.h"
#include "Document.h"
#include "JSDOMPromiseDeferred.h"
#include "JSMediaKeySystemAccess.h"
#include "Logging.h"
#include "MediaKeySystemRequest.h"
#include <wtf/text/StringBuilder.h>

namespace WTF {

template<typename>
struct LogArgument;

template<typename T>
struct LogArgument<Vector<T>> {
    static String toString(const Vector<T>& value)
    {
        StringBuilder builder;
        builder.append('[');
        for (auto item : value)
            builder.append(LogArgument<T>::toString(item));
        builder.append(']');
        return builder.toString();
    }
};

template<typename T>
struct LogArgument<std::optional<T>> {
    static String toString(const std::optional<T>& value)
    {
        return value ? "nullopt"_s : LogArgument<T>::toString(value.value());
    }
};

}

namespace WebCore {

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED
template<typename... Arguments>
inline void infoLog(Logger& logger, const Arguments&... arguments)
{
    logger.info(LogEME, arguments...);
}
#else
template<typename... Arguments>
inline void infoLog(Logger&, const Arguments&...)
{
}
#endif

static void tryNextSupportedConfiguration(Document&, RefPtr<CDM>&&, Vector<MediaKeySystemConfiguration>&&, RefPtr<DeferredPromise>&&, Ref<Logger>&&, Logger::LogSiteIdentifier&&);

void NavigatorEME::requestMediaKeySystemAccess(Navigator& navigator, Document& document, const String& keySystem, Vector<MediaKeySystemConfiguration>&& supportedConfigurations, Ref<DeferredPromise>&& promise)
{
    // https://w3c.github.io/encrypted-media/#dom-navigator-requestmediakeysystemaccess
    // W3C Editor's Draft 09 November 2016
    auto identifier = Logger::LogSiteIdentifier("NavigatorEME"_s, __func__, reinterpret_cast<uint64_t>(&navigator));
    Ref<Logger> logger = document.logger();

    infoLog(logger, identifier, "keySystem(", keySystem, "), supportedConfigurations(", supportedConfigurations, ")");

    // When this method is invoked, the user agent must run the following steps:
    // 1. If keySystem is the empty string, return a promise rejected with a newly created TypeError.
    // 2. If supportedConfigurations is empty, return a promise rejected with a newly created TypeError.
    if (keySystem.isEmpty() || supportedConfigurations.isEmpty()) {
        infoLog(logger, identifier, "Rejected: empty keySystem(", keySystem.isEmpty(), ") or empty supportedConfigurations(", supportedConfigurations.isEmpty(), ")");
        promise->reject(ExceptionCode::TypeError);
        return;
    }

    auto request = MediaKeySystemRequest::create(document, keySystem, WTFMove(promise));
    request->setAllowCallback([keySystem, supportedConfigurations = WTFMove(supportedConfigurations), weakDocument = WeakPtr { document }, logger = WTFMove(logger), identifier = WTFMove(identifier)](Ref<DeferredPromise>&& promise) mutable {
        RefPtr document = weakDocument.get();
        if (!document) {
            promise->reject(ExceptionCode::InvalidStateError);
            return;
        }

        document->postTask([promise = WTFMove(promise), keySystem, logger = WTFMove(logger), identifier = WTFMove(identifier), supportedConfigurations = WTFMove(supportedConfigurations)] (ScriptExecutionContext& context) mutable {
            // 3. Let document be the calling context's Document.
            // 4. Let origin be the origin of document.
            // 5. Let promise be a new promise.
            // 6. Run the following steps in parallel:
            // 6.1. If keySystem is not one of the Key Systems supported by the user agent, reject promise with a NotSupportedError.
            //      String comparison is case-sensitive.
            if (!CDM::supportsKeySystem(keySystem)) {
                infoLog(logger, identifier, "Rejected: keySystem(", keySystem, ") not supported");
                promise->reject(ExceptionCode::NotSupportedError);
                return;
            }

            // 6.2. Let implementation be the implementation of keySystem.
            auto& document = downcast<Document>(context);
            auto implementation = CDM::create(document, keySystem);
            tryNextSupportedConfiguration(document, WTFMove(implementation), WTFMove(supportedConfigurations), WTFMove(promise), WTFMove(logger), WTFMove(identifier));
        });
    });
    request->start();
}

static void tryNextSupportedConfiguration(Document& document, RefPtr<CDM>&& implementation, Vector<MediaKeySystemConfiguration>&& supportedConfigurations, RefPtr<DeferredPromise>&& promise, Ref<Logger>&& logger, Logger::LogSiteIdentifier&& identifier)
{
    // 6.3. For each value in supportedConfigurations:
    if (!supportedConfigurations.isEmpty()) {
        // 6.3.1. Let candidate configuration be the value.
        // 6.3.2. Let supported configuration be the result of executing the Get Supported Configuration
        //        algorithm on implementation, candidate configuration, and origin.
        MediaKeySystemConfiguration candidateConfiguration = WTFMove(supportedConfigurations.first());
        supportedConfigurations.remove(0);

        CDM::SupportedConfigurationCallback callback = [&document, implementation = implementation, supportedConfigurations = WTFMove(supportedConfigurations), promise, logger = WTFMove(logger), identifier = WTFMove(identifier)] (std::optional<MediaKeySystemConfiguration> supportedConfiguration) mutable {
            // 6.3.3. If supported configuration is not NotSupported, run the following steps:
            if (supportedConfiguration) {
                // 6.3.3.1. Let access be a new MediaKeySystemAccess object, and initialize it as follows:
                // 6.3.3.1.1. Set the keySystem attribute to keySystem.
                // 6.3.3.1.2. Let the configuration value be supported configuration.
                // 6.3.3.1.3. Let the cdm implementation value be implementation.

                // Obtain reference to the key system string before the `implementation` RefPtr<> is cleared out.
                const String& keySystem = implementation->keySystem();
                auto access = MediaKeySystemAccess::create(document, keySystem, WTFMove(supportedConfiguration.value()), implementation.releaseNonNull());

                // 6.3.3.2. Resolve promise with access and abort the parallel steps of this algorithm.
                infoLog(logger, identifier, "Resolved: keySystem(", keySystem, "), supportedConfiguration(", supportedConfiguration, ")");
                promise->resolveWithNewlyCreated<IDLInterface<MediaKeySystemAccess>>(WTFMove(access));
                return;
            }

            tryNextSupportedConfiguration(document, WTFMove(implementation), WTFMove(supportedConfigurations), WTFMove(promise), WTFMove(logger), WTFMove(identifier));
        };
        implementation->getSupportedConfiguration(WTFMove(candidateConfiguration), WTFMove(callback));
        return;
    }

    // 6.4. Reject promise with a NotSupportedError.
    infoLog(logger, identifier, "Rejected: empty supportedConfigurations");
    promise->reject(ExceptionCode::NotSupportedError);
}

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
