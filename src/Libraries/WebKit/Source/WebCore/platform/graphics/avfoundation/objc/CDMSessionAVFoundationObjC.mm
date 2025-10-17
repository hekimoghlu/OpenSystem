/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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
#import "CDMSessionAVFoundationObjC.h"

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#import "LegacyCDM.h"
#import "LegacyCDMSession.h"
#import "Logging.h"
#import "MediaPlayer.h"
#import "MediaPlayerPrivateAVFoundationObjC.h"
#import "WebCoreNSErrorExtras.h"
#import <AVFoundation/AVAsset.h>
#import <AVFoundation/AVAssetResourceLoader.h>
#import <JavaScriptCore/TypedArrayInlines.h>
#import <objc/objc-runtime.h>
#import <wtf/LoggerHelper.h>
#import <wtf/MainThread.h>
#import <wtf/SoftLinking.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/UUID.h>
#import <wtf/cocoa/SpanCocoa.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CDMSessionAVFoundationObjC);

CDMSessionAVFoundationObjC::CDMSessionAVFoundationObjC(MediaPlayerPrivateAVFoundationObjC* parent, LegacyCDMSessionClient& client)
    : m_parent(*parent)
    , m_client(client)
    , m_sessionId(createVersion4UUIDString())
#if !RELEASE_LOG_DISABLED
    , m_logger(client.logger())
    , m_logIdentifier(client.logIdentifier())
#endif
{
    ALWAYS_LOG(LOGIDENTIFIER);
}

CDMSessionAVFoundationObjC::~CDMSessionAVFoundationObjC()
{
    ALWAYS_LOG(LOGIDENTIFIER);
}

RefPtr<Uint8Array> CDMSessionAVFoundationObjC::generateKeyRequest(const String& mimeType, Uint8Array* initData, String& destinationURL, unsigned short& errorCode, uint32_t& systemCode)
{
    UNUSED_PARAM(mimeType);

    RefPtr parent = m_parent.get();
    if (!parent) {
        ERROR_LOG(LOGIDENTIFIER, "error: !parent");
        errorCode = LegacyCDM::UnknownError;
        return nullptr;
    }

    String keyURI;
    String keyID;
    RefPtr<Uint8Array> certificate;
    if (!MediaPlayerPrivateAVFoundationObjC::extractKeyURIKeyIDAndCertificateFromInitData(initData, keyURI, keyID, certificate)) {
        ERROR_LOG(LOGIDENTIFIER, "error: could not extract key info");
        errorCode = LegacyCDM::UnknownError;
        return nullptr;
    }

    m_request = parent->takeRequestForKeyURI(keyURI);
    if (!m_request) {
        ERROR_LOG(LOGIDENTIFIER, "error: could not find request for key URI");
        errorCode = LegacyCDM::UnknownError;
        return nullptr;
    }

    RetainPtr certificateData = toNSData(certificate->span());
    NSString* assetStr = keyID;
    RetainPtr<NSData> assetID = [assetStr dataUsingEncoding:NSUTF8StringEncoding];
    NSError* nsError = 0;
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    RetainPtr<NSData> keyRequest = [m_request streamingContentKeyRequestDataForApp:certificateData.get() contentIdentifier:assetID.get() options:nil error:&nsError];
ALLOW_DEPRECATED_DECLARATIONS_END

    if (!keyRequest) {
        ERROR_LOG(LOGIDENTIFIER, "failed to generate key request with error: ", nsError);
        errorCode = LegacyCDM::DomainError;
        systemCode = mediaKeyErrorSystemCode(nsError);
        return nullptr;
    }

    errorCode = MediaPlayer::NoError;
    systemCode = 0;
    destinationURL = String();

    ALWAYS_LOG(LOGIDENTIFIER);

    return Uint8Array::create(ArrayBuffer::create(span(keyRequest.get())));
}

void CDMSessionAVFoundationObjC::releaseKeys()
{
}

bool CDMSessionAVFoundationObjC::update(Uint8Array* key, RefPtr<Uint8Array>& nextMessage, unsigned short& errorCode, uint32_t& systemCode)
{
    RetainPtr keyData = toNSData(key->span());
    [[m_request dataRequest] respondWithData:keyData.get()];
    [m_request finishLoading];
    errorCode = MediaPlayer::NoError;
    systemCode = 0;
    nextMessage = nullptr;

    ALWAYS_LOG(LOGIDENTIFIER);

    return true;
}

RefPtr<ArrayBuffer> CDMSessionAVFoundationObjC::cachedKeyForKeyID(const String&) const
{
    return nullptr;
}

void CDMSessionAVFoundationObjC::playerDidReceiveError(NSError *error)
{
    if (!m_client)
        return;

    ERROR_LOG(LOGIDENTIFIER, error);

    unsigned long code = mediaKeyErrorSystemCode(error);
    m_client->sendError(LegacyCDMSessionClient::MediaKeyErrorDomain, code);
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& CDMSessionAVFoundationObjC::logChannel() const
{
    return LogEME;
}
#endif

}

#endif
