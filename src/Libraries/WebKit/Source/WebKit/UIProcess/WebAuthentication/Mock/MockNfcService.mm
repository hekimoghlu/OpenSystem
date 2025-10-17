/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
#import "MockNfcService.h"

#if ENABLE(WEB_AUTHN)
#import "CtapNfcDriver.h"
#import "NearFieldSPI.h"
#import "NfcConnection.h"
#import <WebCore/FidoConstants.h>
#import <wtf/BlockPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/RunLoop.h>
#import <wtf/Vector.h>

#import "NearFieldSoftLink.h"

namespace {
uint8_t tagID1[] = { 0x01 };
uint8_t tagID2[] = { 0x02 };
}

#if HAVE(NEAR_FIELD)

@interface WKMockNFTag : NSObject <NFTag>

- (instancetype)initWithType:(NFTagType)type;
- (instancetype)initWithType:(NFTagType)type tagID:(NSData *)tagID;

@end

@implementation WKMockNFTag {
    NFTagType _type;
    RetainPtr<NSData> _tagID;
}

@synthesize technology = _technology;
@synthesize AppData = _AppData;
@synthesize UID = _UID;
@synthesize ndefAvailability = _ndefAvailability;
@synthesize ndefMessageSize = _ndefMessageSize;
@synthesize ndefContainerSize = _ndefContainerSize;
@synthesize tagA = _tagA;
@synthesize tagB = _tagB;
@synthesize tagF = _tagF;

- (NFTagType)type
{
    return _type;
}

- (NSData *)tagID
{
    return _tagID.get();
}

- (instancetype)initWithNFTag:(id<NFTag>)tag
{
    if ((self = [super init])) {
        _type = tag.type;
        _tagID = tag.tagID;
    }
    return self;
}

- (void)dealloc
{
    [_AppData release];
    _AppData = nil;
    [_UID release];
    _UID = nil;

    [super dealloc];
}

- (NSString*)description
{
    return nil;
}

- (BOOL)isEqualToNFTag:(id<NFTag>)tag
{
    return NO;
}

- (instancetype)initWithType:(NFTagType)type
{
    return [self initWithType:type tagID:adoptNS([[NSData alloc] initWithBytesNoCopy:tagID1 length:sizeof(tagID1) freeWhenDone:NO]).get()];
}

- (instancetype)initWithType:(NFTagType)type tagID:(NSData *)tagID
{
    if ((self = [super init])) {
        _type = type;
        _tagID = tagID;
    }
    return self;
}

@end

#endif // HAVE(NEAR_FIELD)

namespace WebKit {
using namespace fido;
using Mock = WebCore::MockWebAuthenticationConfiguration;

#if HAVE(NEAR_FIELD)

namespace {

static id<NFReaderSessionDelegate> globalNFReaderSessionDelegate;
static MockNfcService* globalNfcService;

static void NFReaderSessionSetDelegate(id, SEL, id<NFReaderSessionDelegate> delegate)
{
    globalNFReaderSessionDelegate = delegate;
}

static BOOL NFReaderSessionConnectTagFail(id, SEL, NFTag *)
{
    return NO;
}

static BOOL NFReaderSessionConnectTag(id, SEL, NFTag *)
{
    return YES;
}

static BOOL NFReaderSessionStopPolling(id, SEL)
{
    if (!globalNfcService)
        return NO;
    globalNfcService->receiveStopPolling();
    return YES;
}

static BOOL NFReaderSessionStartPollingWithError(id, SEL, NSError **)
{
    if (!globalNfcService)
        return NO;
    globalNfcService->receiveStartPolling();
    return YES;
}

static NSData* NFReaderSessionTransceive(id, SEL, NSData *)
{
    if (!globalNfcService)
        return nil;
    return globalNfcService->transceive();
}

} // namespace

#endif // HAVE(NEAR_FIELD)

Ref<MockNfcService> MockNfcService::create(AuthenticatorTransportServiceObserver& observer, const WebCore::MockWebAuthenticationConfiguration& configuration)
{
    return adoptRef(*new MockNfcService(observer, configuration));
}

MockNfcService::MockNfcService(AuthenticatorTransportServiceObserver& observer, const WebCore::MockWebAuthenticationConfiguration& configuration)
    : NfcService(observer)
    , m_configuration(configuration)
{
}

NSData* MockNfcService::transceive()
{
    if (m_configuration.nfc->payloadBase64.isEmpty())
        return nil;

    auto result = adoptNS([[NSData alloc] initWithBase64EncodedString:m_configuration.nfc->payloadBase64[0] options:NSDataBase64DecodingIgnoreUnknownCharacters]);
    m_configuration.nfc->payloadBase64.remove(0);
    return result.autorelease();
}

void MockNfcService::receiveStopPolling()
{
    // For purpose of restart polling.
    m_configuration.nfc->multiplePhysicalTags = false;
}

void MockNfcService::receiveStartPolling()
{
    RunLoop::main().dispatch([weakThis = WeakPtr { *this }] {
        if (!weakThis)
            return;
        weakThis->detectTags();
    });
}

void MockNfcService::platformStartDiscovery()
{
#if HAVE(NEAR_FIELD)
    if (!!m_configuration.nfc) {
        globalNfcService = this;

        Method methodToSwizzle1 = class_getInstanceMethod(getNFReaderSessionClass(), @selector(setDelegate:));
        method_setImplementation(methodToSwizzle1, (IMP)NFReaderSessionSetDelegate);

        Method methodToSwizzle2 = class_getInstanceMethod(getNFReaderSessionClass(), @selector(connectTag:));
        if (m_configuration.nfc->error == Mock::NfcError::NoConnections)
            method_setImplementation(methodToSwizzle2, (IMP)NFReaderSessionConnectTagFail);
        else
            method_setImplementation(methodToSwizzle2, (IMP)NFReaderSessionConnectTag);

        Method methodToSwizzle3 = class_getInstanceMethod(getNFReaderSessionClass(), @selector(transceive:));
        method_setImplementation(methodToSwizzle3, (IMP)NFReaderSessionTransceive);

        Method methodToSwizzle4 = class_getInstanceMethod(getNFReaderSessionClass(), @selector(stopPolling));
        method_setImplementation(methodToSwizzle4, (IMP)NFReaderSessionStopPolling);

        Method methodToSwizzle5 = class_getInstanceMethod(getNFReaderSessionClass(), @selector(startPollingWithError:));
        method_setImplementation(methodToSwizzle5, (IMP)NFReaderSessionStartPollingWithError);

        auto readerSession = adoptNS([allocNFReaderSessionInstance() initWithUIType:NFReaderSessionUINone]);
        setConnection(NfcConnection::create(readerSession.get(), *this));
    }
    LOG_ERROR("No nfc authenticators is available.");
#endif // HAVE(NEAR_FIELD)
}

void MockNfcService::detectTags() const
{
#if HAVE(NEAR_FIELD)
    if (m_configuration.nfc->error == Mock::NfcError::NoTags)
        return;

    auto callback = makeBlockPtr([configuration = m_configuration] {
        auto tags = adoptNS([[NSMutableArray alloc] init]);
        if (configuration.nfc->error == Mock::NfcError::WrongTagType || configuration.nfc->multipleTags)
            [tags addObject:adoptNS([[WKMockNFTag alloc] initWithType:NFTagTypeUnknown]).get()];
        else
            [tags addObject:adoptNS([[WKMockNFTag alloc] initWithType:NFTagTypeGeneric4A]).get()];

        if (configuration.nfc->multipleTags)
            [tags addObject:adoptNS([[WKMockNFTag alloc] initWithType:NFTagTypeGeneric4A]).get()];

        if (configuration.nfc->multiplePhysicalTags)
            [tags addObject:adoptNS([[WKMockNFTag alloc] initWithType:NFTagTypeGeneric4A tagID:adoptNS([[NSData alloc] initWithBytesNoCopy:tagID2 length:sizeof(tagID2) freeWhenDone:NO]).get()]).get()];

        auto readerSession = adoptNS([allocNFReaderSessionInstance() initWithUIType:NFReaderSessionUINone]);
        [globalNFReaderSessionDelegate readerSession:readerSession.get() didDetectTags:tags.get()];
    });
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), callback.get());
#endif // HAVE(NEAR_FIELD)
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
