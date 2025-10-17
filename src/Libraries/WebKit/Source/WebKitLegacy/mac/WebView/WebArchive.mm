/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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
#import "WebArchive.h"
#import "WebArchiveInternal.h"

#import "WebKitLogging.h"
#import "WebNSObjectExtras.h"
#import "WebResourceInternal.h"
#import <JavaScriptCore/InitializeThreading.h>
#import <WebCore/ArchiveResource.h>
#import <WebCore/LegacyWebArchive.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreJITOperations.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/MainThread.h>
#import <wtf/RunLoop.h>
#import <wtf/cocoa/VectorCocoa.h>

using namespace WebCore;

NSString *WebArchivePboardType = @"Apple Web Archive pasteboard type";

static NSString * const WebMainResourceKey = @"WebMainResource";
static NSString * const WebSubresourcesKey = @"WebSubresources";
static NSString * const WebSubframeArchivesKey = @"WebSubframeArchives";

@interface WebArchivePrivate : NSObject {
@public
    RetainPtr<WebResource> cachedMainResource;
    RetainPtr<NSArray> cachedSubresources;
    RetainPtr<NSArray> cachedSubframeArchives;
@private
    RefPtr<LegacyWebArchive> coreArchive;
}

- (instancetype)initWithCoreArchive:(RefPtr<LegacyWebArchive>&&)coreArchive;
- (LegacyWebArchive*)coreArchive;
- (void)setCoreArchive:(Ref<LegacyWebArchive>&&)newCoreArchive;
@end

@implementation WebArchivePrivate

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
    if (!self)
        return nil;
    coreArchive = LegacyWebArchive::create();
    return self;
}

- (instancetype)initWithCoreArchive:(RefPtr<LegacyWebArchive>&&)_coreArchive
{
    self = [super init];
    if (!self|| !_coreArchive) {
        [self release];
        return nil;
    }
    coreArchive = WTFMove(_coreArchive);
    return self;
}

- (LegacyWebArchive*)coreArchive
{
    return coreArchive.get();
}

- (void)setCoreArchive:(Ref<LegacyWebArchive>&&)newCoreArchive
{
    ASSERT(coreArchive);
    coreArchive = WTFMove(newCoreArchive);
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([WebArchivePrivate class], self))
        return;
    
    [super dealloc];
}

@end

@implementation WebArchive

- (instancetype)init
{
    WebCoreThreadViolationCheckRoundTwo();

    self = [super init];
    if (!self)
        return nil;
    _private = [[WebArchivePrivate alloc] init];
    return self;
}

static BOOL isArrayOfClass(id object, Class elementClass)
{
    if (![object isKindOfClass:[NSArray class]])
        return NO;
    NSArray *array = (NSArray *)object;
    NSUInteger count = [array count];
    for (NSUInteger i = 0; i < count; ++i)
        if (![[array objectAtIndex:i] isKindOfClass:elementClass])
            return NO;
    return YES;
}

- (instancetype)initWithMainResource:(WebResource *)mainResource subresources:(NSArray *)subresources subframeArchives:(NSArray *)subframeArchives
{
    WebCoreThreadViolationCheckRoundTwo();

    self = [super init];
    if (!self)
        return nil;

    _private = [[WebArchivePrivate alloc] init];

    _private->cachedMainResource = mainResource;
    if (!_private->cachedMainResource) {
        [self release];
        return nil;
    }
    
    if (!subresources || isArrayOfClass(subresources, [WebResource class]))
        _private->cachedSubresources = subresources;
    else {
        [self release];
        return nil;
    }

    if (!subframeArchives || isArrayOfClass(subframeArchives, [WebArchive class]))
        _private->cachedSubframeArchives = subframeArchives;
    else {
        [self release];
        return nil;
    }

    Vector<Ref<ArchiveResource>> coreResources;
    for (WebResource *subresource in subresources)
        coreResources.append([subresource _coreResource].get());

    Vector<Ref<LegacyWebArchive>> coreArchives;
    for (WebArchive *subframeArchive in subframeArchives)
        coreArchives.append(*[subframeArchive->_private coreArchive]);

    [_private setCoreArchive:LegacyWebArchive::create([mainResource _coreResource].get(), WTFMove(coreResources), WTFMove(coreArchives))];
    return self;
}

- (instancetype)initWithData:(NSData *)data
{
    WebCoreThreadViolationCheckRoundTwo();

    self = [super init];
    if (!self)
        return nil;
        
#if !LOG_DISABLED
    CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
#endif

    _private = [[WebArchivePrivate alloc] init];
    auto coreArchive = LegacyWebArchive::create(SharedBuffer::create(data));
    if (!coreArchive) {
        [self release];
        return nil;
    }
        
    [_private setCoreArchive:coreArchive.releaseNonNull()];
        
#if !LOG_DISABLED
    CFAbsoluteTime end = CFAbsoluteTimeGetCurrent();
    CFAbsoluteTime duration = end - start;
#endif
    LOG(Timing, "Parsing web archive with [NSPropertyListSerialization propertyListFromData::::] took %f seconds", duration);
    
    return self;
}

- (instancetype)initWithCoder:(NSCoder *)decoder
{    
    WebResource *mainResource = nil;
    NSArray *subresources = nil;
    NSArray *subframeArchives = nil;
    
    @try {
        id object = [decoder decodeObjectForKey:WebMainResourceKey];
        if ([object isKindOfClass:[WebResource class]])
            mainResource = object;
        object = [decoder decodeObjectForKey:WebSubresourcesKey];
        if (isArrayOfClass(object, [WebResource class]))
            subresources = object;
        object = [decoder decodeObjectForKey:WebSubframeArchivesKey];
        if (isArrayOfClass(object, [WebArchive class]))
            subframeArchives = object;
    } @catch(id) {
        [self release];
        return nil;
    }

    return [self initWithMainResource:mainResource subresources:subresources subframeArchives:subframeArchives];
}

- (void)encodeWithCoder:(NSCoder *)encoder
{
    [encoder encodeObject:[self mainResource] forKey:WebMainResourceKey];
    [encoder encodeObject:[self subresources] forKey:WebSubresourcesKey];
    [encoder encodeObject:[self subframeArchives] forKey:WebSubframeArchivesKey];    
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

- (WebResource *)mainResource
{
    WebCoreThreadViolationCheckRoundTwo();

    // Currently from WebKit API perspective, WebArchives are entirely immutable once created
    // If they ever become mutable, we'll need to rethink this. 
    if (!_private->cachedMainResource) {
        if (auto* coreArchive = [_private coreArchive]) {
            if (auto* mainResource = coreArchive->mainResource())
                _private->cachedMainResource = adoptNS([[WebResource alloc] _initWithCoreResource:*mainResource]);
        }
    }
    
    auto cachedMainResourceCopy = _private->cachedMainResource;
    return cachedMainResourceCopy.autorelease();
}

- (NSArray *)subresources
{
    WebCoreThreadViolationCheckRoundTwo();

    // Currently from WebKit API perspective, WebArchives are entirely immutable once created
    // If they ever become mutable, we'll need to rethink this.     
    if (!_private->cachedSubresources) {
        auto coreArchive = [_private coreArchive];
        if (!coreArchive)
            _private->cachedSubresources = adoptNS([[NSArray alloc] init]);
        else {
            _private->cachedSubresources = createNSArray(coreArchive->subresources(), [] (auto& subresource) {
                return adoptNS([[WebResource alloc] _initWithCoreResource:subresource.get()]);
            });
        }
    }
    // Maintain the WebKit 3 behavior of this API, which is documented and
    // relied upon by some clients, of returning nil if there are no subresources.
    if (![_private->cachedSubresources count])
        return nil;

    auto cachedSubresourcesCopy = _private->cachedSubresources;
    return cachedSubresourcesCopy.autorelease();
}

- (NSArray *)subframeArchives
{
    WebCoreThreadViolationCheckRoundTwo();

    // Currently from WebKit API perspective, WebArchives are entirely immutable once created
    // If they ever become mutable, we'll need to rethink this.  
    if (!_private->cachedSubframeArchives) {
        auto* coreArchive = [_private coreArchive];
        if (!coreArchive)
            _private->cachedSubframeArchives = adoptNS([[NSArray alloc] init]);
        else {
            _private->cachedSubframeArchives = createNSArray(coreArchive->subframeArchives(), [] (auto& archive) {
                return adoptNS([[WebArchive alloc] _initWithCoreLegacyWebArchive:static_cast<LegacyWebArchive*>(archive.ptr())]);
            });
        }
    }
    
    auto cachedSubframeArchivesCopy = _private->cachedSubframeArchives;
    return cachedSubframeArchivesCopy.autorelease();
}

- (NSData *)data
{
    WebCoreThreadViolationCheckRoundTwo();

#if !LOG_DISABLED
    CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
#endif

    RetainPtr<CFDataRef> data = [_private coreArchive]->rawDataRepresentation();
    
#if !LOG_DISABLED
    CFAbsoluteTime end = CFAbsoluteTimeGetCurrent();
    CFAbsoluteTime duration = end - start;
#endif
    LOG(Timing, "Serializing web archive to raw CFPropertyList data took %f seconds", duration);
        
    return retainPtr((NSData *)data.get()).autorelease();
}

@end

@implementation WebArchive (WebInternal)

- (id)_initWithCoreLegacyWebArchive:(RefPtr<WebCore::LegacyWebArchive>&&)coreLegacyWebArchive
{
    WebCoreThreadViolationCheckRoundTwo();

    self = [super init];
    if (!self)
        return nil;
    
    _private = [[WebArchivePrivate alloc] initWithCoreArchive:WTFMove(coreLegacyWebArchive)];
    if (!_private) {
        [self release];
        return nil;
    }

    return self;
}

- (WebCore::LegacyWebArchive *)_coreLegacyWebArchive
{
    WebCoreThreadViolationCheckRoundTwo();

    return [_private coreArchive];
}

@end
