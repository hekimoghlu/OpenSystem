/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#import "WebResourceInternal.h"

#import "WebFrameInternal.h"
#import "WebKitLogging.h"
#import "WebKitVersionChecks.h"
#import "WebNSDictionaryExtras.h"
#import "WebNSObjectExtras.h"
#import "WebNSURLExtras.h"
#import <JavaScriptCore/InitializeThreading.h>
#import <WebCore/ArchiveResource.h>
#import <WebCore/LegacyWebArchive.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreJITOperations.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebCoreURLResponse.h>
#import <pal/text/TextEncoding.h>
#import <wtf/MainThread.h>
#import <wtf/RefPtr.h>
#import <wtf/RunLoop.h>
#import <wtf/RuntimeApplicationChecks.h>

using namespace WebCore;

static NSString * const WebResourceDataKey =              @"WebResourceData";
static NSString * const WebResourceFrameNameKey =         @"WebResourceFrameName";
static NSString * const WebResourceMIMETypeKey =          @"WebResourceMIMEType";
static NSString * const WebResourceURLKey =               @"WebResourceURL";
static NSString * const WebResourceTextEncodingNameKey =  @"WebResourceTextEncodingName";
static NSString * const WebResourceResponseKey =          @"WebResourceResponse";

@interface WebResourcePrivate : NSObject {
@public
    RefPtr<ArchiveResource> coreResource;
}
- (instancetype)initWithCoreResource:(Ref<ArchiveResource>&&)coreResource;
@end

@implementation WebResourcePrivate

+ (void)initialize
{
#if !PLATFORM(IOS_FAMILY)
    JSC::initialize();
    WTF::initializeMainThread();
    WebCore::populateJITOperations();
#endif
}

- (instancetype)init
{
    self = [super init];
    return self;
}

- (instancetype)initWithCoreResource:(Ref<ArchiveResource>&&)passedResource
{
    self = [super init];
    if (!self)
        return nil;
    coreResource = WTFMove(passedResource);
    return self;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([WebResourcePrivate class], self))
        return;
    [super dealloc];
}

@end

@implementation WebResource

- (instancetype)init
{
    self = [super init];
    if (!self)
        return nil;
    _private = [[WebResourcePrivate alloc] init];
    return self;
}

- (instancetype)initWithData:(NSData *)data URL:(NSURL *)URL MIMEType:(NSString *)MIMEType textEncodingName:(NSString *)textEncodingName frameName:(NSString *)frameName
{
    return [self _initWithData:data URL:URL MIMEType:MIMEType textEncodingName:textEncodingName frameName:frameName response:nil copyData:YES];
}

- (instancetype)initWithCoder:(NSCoder *)decoder
{
    WebCoreThreadViolationCheckRoundTwo();

    self = [super init];
    if (!self)
        return nil;

    NSData *data = nil;
    NSURL *url = nil;
    NSString *mimeType = nil, *textEncoding = nil, *frameName = nil;
    NSURLResponse *response = nil;
    
    @try {
        id object = [decoder decodeObjectForKey:WebResourceDataKey];
        if ([object isKindOfClass:[NSData class]])
            data = object;
        object = [decoder decodeObjectForKey:WebResourceURLKey];
        if ([object isKindOfClass:[NSURL class]])
            url = object;
        object = [decoder decodeObjectForKey:WebResourceMIMETypeKey];
        if ([object isKindOfClass:[NSString class]])
            mimeType = object;
        object = [decoder decodeObjectForKey:WebResourceTextEncodingNameKey];
        if ([object isKindOfClass:[NSString class]])
            textEncoding = object;
        object = [decoder decodeObjectForKey:WebResourceFrameNameKey];
        if ([object isKindOfClass:[NSString class]])
            frameName = object;
        object = [decoder decodeObjectForKey:WebResourceResponseKey];
        if ([object isKindOfClass:[NSURLResponse class]])
            response = object;
    } @catch(id) {
        [self release];
        return nil;
    }

    auto coreResource = ArchiveResource::create(SharedBuffer::create(data), url, mimeType, textEncoding, frameName, response);
    if (!coreResource) {
        [self release];
        return nil;
    }

    _private = [[WebResourcePrivate alloc] initWithCoreResource:coreResource.releaseNonNull()];
    return self;
}

- (void)encodeWithCoder:(NSCoder *)encoder
{
    auto* resource = _private->coreResource.get();

    RetainPtr<NSData> data;
    NSURL *url = nil;
    NSString *mimeType = nil, *textEncoding = nil, *frameName = nil;
    NSURLResponse *response = nil;

    if (resource) {
        data = resource->data().makeContiguous()->createNSData();
        url = resource->url();
        mimeType = resource->mimeType();
        textEncoding = resource->textEncoding();
        frameName = resource->frameName();
        response = resource->response().nsURLResponse();
    }
    [encoder encodeObject:data.get() forKey:WebResourceDataKey];
    [encoder encodeObject:url forKey:WebResourceURLKey];
    [encoder encodeObject:mimeType forKey:WebResourceMIMETypeKey];
    [encoder encodeObject:textEncoding forKey:WebResourceTextEncodingNameKey];
    [encoder encodeObject:frameName forKey:WebResourceFrameNameKey];
    [encoder encodeObject:response forKey:WebResourceResponseKey];
}

- (void)dealloc
{
    [_private release];
    [super dealloc];
}

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

- (NSData *)data
{
    WebCoreThreadViolationCheckRoundTwo();

    if (!_private->coreResource)
        return nil;
    return _private->coreResource->data().makeContiguous()->createNSData().autorelease();
}

- (NSURL *)URL
{
    WebCoreThreadViolationCheckRoundTwo();

    if (!_private->coreResource)
        return nil;
    return _private->coreResource->url();
}

- (NSString *)MIMEType
{
    WebCoreThreadViolationCheckRoundTwo();

    if (!_private->coreResource)
        return nil;
    NSString *mimeType = _private->coreResource->mimeType();
    return mimeType;
}

- (NSString *)textEncodingName
{
    WebCoreThreadViolationCheckRoundTwo();

    if (!_private->coreResource)
        return nil;
    NSString *textEncodingName = _private->coreResource->textEncoding();
    return textEncodingName;
}

- (NSString *)frameName
{
    WebCoreThreadViolationCheckRoundTwo();

    if (!_private->coreResource)
        return nil;
    NSString *frameName = _private->coreResource->frameName();
    return frameName;
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@ %@>", NSStringFromClass(self.class), self.URL];
}

@end

@implementation WebResource (WebResourceInternal)

- (id)_initWithCoreResource:(Ref<ArchiveResource>&&)coreResource
{
    self = [super init];
    if (!self)
        return nil;

    _private = [[WebResourcePrivate alloc] initWithCoreResource:WTFMove(coreResource)];
    return self;
}

- (NakedRef<WebCore::ArchiveResource>)_coreResource
{
    return *_private->coreResource;
}

@end

@implementation WebResource (WebResourcePrivate)

// SPI for Mail (5066325)
// FIXME: This "ignoreWhenUnarchiving" concept is an ugly one - can we find a cleaner solution for those who need this SPI?
- (void)_ignoreWhenUnarchiving
{
    WebCoreThreadViolationCheckRoundTwo();

    if (!_private->coreResource)
        return;
    _private->coreResource->ignoreWhenUnarchiving();
}

- (id)_initWithData:(NSData *)data 
                URL:(NSURL *)URL 
           MIMEType:(NSString *)MIMEType 
   textEncodingName:(NSString *)textEncodingName 
          frameName:(NSString *)frameName 
           response:(NSURLResponse *)response
           copyData:(BOOL)copyData
{
    WebCoreThreadViolationCheckRoundTwo();

    self = [super init];
    if (!self)
        return nil;
    
    if (!data || !URL || !MIMEType) {
        [self release];
        return nil;
    }

    auto coreResource = ArchiveResource::create(SharedBuffer::create(copyData ? adoptNS([data copy]).get() : data), URL, MIMEType, textEncodingName, frameName, response);
    if (!coreResource) {
        [self release];
        return nil;
    }

    _private = [[WebResourcePrivate alloc] initWithCoreResource:coreResource.releaseNonNull()];
    return self;
}

- (id)_initWithData:(NSData *)data URL:(NSURL *)URL response:(NSURLResponse *)response
{
    // Pass NO for copyData since the data doesn't need to be copied since we know that callers will no longer modify it.
    // Copying it will also cause a performance regression.
    return [self _initWithData:data
                           URL:URL
                      MIMEType:[response MIMEType]
              textEncodingName:[response textEncodingName]
                     frameName:nil
                      response:response
                      copyData:NO];
}

- (NSString *)_suggestedFilename
{
    WebCoreThreadViolationCheckRoundTwo();

    if (!_private->coreResource)
        return nil;
    NSString *suggestedFilename = _private->coreResource->response().suggestedFilename();
    return suggestedFilename;
}

#if !PLATFORM(IOS_FAMILY)
- (NSFileWrapper *)_fileWrapperRepresentation
{
    auto wrapper = adoptNS([[NSFileWrapper alloc] initRegularFileWithContents:[self data]]);
    NSString *filename = [self _suggestedFilename];
    if (!filename || ![filename length])
        filename = [[self URL] _webkit_suggestedFilenameWithMIMEType:[self MIMEType]];
    [wrapper setPreferredFilename:filename];
    return wrapper.autorelease();
}
#endif

- (NSURLResponse *)_response
{
    WebCoreThreadViolationCheckRoundTwo();

    NSURLResponse *response = nil;
    if (_private->coreResource)
        response = _private->coreResource->response().nsURLResponse();
    return response ? response : adoptNS([[NSURLResponse alloc] init]).autorelease();
}

- (NSString *)_stringValue
{
    WebCoreThreadViolationCheckRoundTwo();

    PAL::TextEncoding encoding;
    if (_private->coreResource)
        encoding = _private->coreResource->textEncoding();
    if (!encoding.isValid())
        encoding = PAL::WindowsLatin1Encoding();
    
    RefPtr coreData = _private->coreResource ? &_private->coreResource->data() : nullptr;
    if (!coreData)
        return @"";
    return encoding.decode(coreData->makeContiguous()->span());
}

@end
