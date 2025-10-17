/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
#import "WebBackForwardList.h"
#import "WebBackForwardListInternal.h"

#import "BackForwardList.h"
#import "WebFrameInternal.h"
#import "WebHistoryItemInternal.h"
#import "WebHistoryItemPrivate.h"
#import "WebKitLogging.h"
#import "WebKitVersionChecks.h"
#import "WebNSObjectExtras.h"
#import "WebPreferencesPrivate.h"
#import "WebViewPrivate.h"
#import <JavaScriptCore/InitializeThreading.h>
#import <WebCore/BackForwardCache.h>
#import <WebCore/HistoryItem.h>
#import <WebCore/Settings.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreJITOperations.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/Assertions.h>
#import <wtf/MainThread.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RunLoop.h>
#import <wtf/StdLibExtras.h>
#import <wtf/cocoa/VectorCocoa.h>

using BackForwardListMap = HashMap<WeakRef<BackForwardList>, WebBackForwardList*>;

// FIXME: Instead of this we could just create a class derived from BackForwardList
// with a pointer to a WebBackForwardList in it.
static BackForwardListMap& backForwardLists()
{
    static NeverDestroyed<BackForwardListMap> staticBackForwardLists;
    return staticBackForwardLists;
}

@implementation WebBackForwardList

BackForwardList* core(WebBackForwardList *webBackForwardList)
{
    if (!webBackForwardList)
        return 0;

    return reinterpret_cast<BackForwardList*>(webBackForwardList->_private);
}

WebBackForwardList *kit(BackForwardList* backForwardList)
{
    if (!backForwardList)
        return nil;

    if (WebBackForwardList *webBackForwardList = backForwardLists().get(*backForwardList))
        return webBackForwardList;

    return adoptNS([[WebBackForwardList alloc] initWithBackForwardList:*backForwardList]).autorelease();
}

- (id)initWithBackForwardList:(Ref<BackForwardList>&&)backForwardList
{   
    WebCoreThreadViolationCheckRoundOne();
    self = [super init];
    if (!self)
        return nil;

    _private = reinterpret_cast<WebBackForwardListPrivate*>(&backForwardList.leakRef());
    backForwardLists().set(*core(self), self);
    return self;
}

+ (void)initialize
{
#if !PLATFORM(IOS_FAMILY)
    JSC::initialize();
    WTF::initializeMainThread();
    WebCore::populateJITOperations();
#endif
}

- (id)init
{
    return [self initWithBackForwardList:BackForwardList::create(nullptr)];
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([WebBackForwardList class], self))
        return;

    BackForwardList* backForwardList = core(self);
    ASSERT(backForwardList);
    if (backForwardList) {
        ASSERT(backForwardList->closed());
        backForwardLists().remove(*backForwardList);
        backForwardList->deref();
    }

    [super dealloc];
}

- (void)_close
{
    core(self)->close();
}

- (void)addItem:(WebHistoryItem *)entry
{
    ASSERT(entry);
    core(self)->addItem(*core(entry));
    
    // Since the assumed contract with WebBackForwardList is that it retains its WebHistoryItems,
    // the following line prevents a whole class of problems where a history item will be created in
    // a function, added to the BFlist, then used in the rest of that function.
    retainPtr(entry).autorelease();
}

- (void)removeItem:(WebHistoryItem *)item
{
    if (!item)
        return;

    core(self)->removeItem(*core(item));
}

#if PLATFORM(IOS_FAMILY)

// FIXME: Move into WebCore the code that deals directly with WebCore::BackForwardList.

constexpr auto WebBackForwardListDictionaryEntriesKey = @"entries";
constexpr auto WebBackForwardListDictionaryCapacityKey = @"capacity";
constexpr auto WebBackForwardListDictionaryCurrentKey = @"current";

- (NSDictionary *)dictionaryRepresentation
{
    auto& list = *core(self);
    auto entries = createNSArray(list.entries(), [] (auto& item) {
        return [kit(const_cast<WebCore::HistoryItem*>(item.ptr())) dictionaryRepresentationIncludingChildren:NO];
    });
    return @{
        WebBackForwardListDictionaryEntriesKey: entries.get(),
        WebBackForwardListDictionaryCurrentKey: @(list.current()),
        WebBackForwardListDictionaryCapacityKey: @(list.capacity()),
    };
}

- (void)setToMatchDictionaryRepresentation:(NSDictionary *)dictionary
{
    auto& list = *core(self);

    list.setCapacity([[dictionary objectForKey:WebBackForwardListDictionaryCapacityKey] unsignedIntValue]);
    for (NSDictionary *itemDictionary in [dictionary objectForKey:WebBackForwardListDictionaryEntriesKey])
        list.addItem(*core(adoptNS([[WebHistoryItem alloc] initFromDictionaryRepresentation:itemDictionary]).get()));

    unsigned currentIndex = [[dictionary objectForKey:WebBackForwardListDictionaryCurrentKey] unsignedIntValue];
    size_t listSize = list.entries().size();
    if (currentIndex >= listSize)
        currentIndex = listSize - 1;
    list.setCurrent(currentIndex);
}

#endif // PLATFORM(IOS_FAMILY)

- (BOOL)containsItem:(WebHistoryItem *)item
{
    if (!item)
        return NO;

    return core(self)->containsItem(*core(item));
}

- (void)goBack
{
    core(self)->goBack();
}

- (void)goForward
{
    core(self)->goForward();
}

- (void)goToItem:(WebHistoryItem *)item
{
    if (item)
        core(self)->goToItem(*core(item));
}

- (WebHistoryItem *)backItem
{
    return retainPtr(kit(core(self)->backItem().get())).autorelease();
}

- (WebHistoryItem *)currentItem
{
    return retainPtr(kit(core(self)->currentItem().get())).autorelease();
}

- (WebHistoryItem *)forwardItem
{
    return retainPtr(kit(core(self)->forwardItem().get())).autorelease();
}

static bool bumperCarBackForwardHackNeeded()
{
#if !PLATFORM(IOS_FAMILY)
    static bool hackNeeded = [[[NSBundle mainBundle] bundleIdentifier] isEqualToString:@"com.freeverse.bumpercar"]
        && !WebKitLinkedOnOrAfter(WEBKIT_FIRST_VERSION_WITHOUT_BUMPERCAR_BACK_FORWARD_QUIRK);
    return hackNeeded;
#else
    return false;
#endif
}

- (NSArray *)backListWithLimit:(int)limit
{
    Vector<Ref<WebCore::HistoryItem>> list;
    core(self)->backListWithLimit(limit, list);
    auto result = createNSArray(list, [] (auto& item) {
        return kit(item.ptr());
    });
    if (bumperCarBackForwardHackNeeded()) {
        static NeverDestroyed<RetainPtr<NSArray>> lastBackListArray;
        lastBackListArray.get() = result;
    }
    return result.autorelease();
}

- (NSArray *)forwardListWithLimit:(int)limit
{
    Vector<Ref<WebCore::HistoryItem>> list;
    core(self)->forwardListWithLimit(limit, list);
    auto result = createNSArray(list, [] (auto& item) {
        return kit(item.ptr());
    });
    if (bumperCarBackForwardHackNeeded()) {
        static NeverDestroyed<RetainPtr<NSArray>> lastForwardListArray;
        lastForwardListArray.get() = result;
    }
    return result.autorelease();
}

- (int)capacity
{
    return core(self)->capacity();
}

- (void)setCapacity:(int)size
{
    core(self)->setCapacity(size);
}


-(NSString *)description
{
    NSMutableString *result;
    
    result = [NSMutableString stringWithCapacity:512];
    
    [result appendString:@"\n--------------------------------------------\n"];    
    [result appendString:@"WebBackForwardList:\n"];
    
    BackForwardList* backForwardList = core(self);
    auto& entries = backForwardList->entries();

    for (unsigned i = 0; i < entries.size(); ++i) {
        if (entries[i].ptr() == backForwardList->currentItem()) {
            [result appendString:@" >>>"]; 
        } else {
            [result appendString:@"    "]; 
        }   
        [result appendFormat:@"%2d) ", i];
        int currPos = [result length];
        [result appendString:[kit(const_cast<WebCore::HistoryItem*>(entries[i].ptr())) description]];

        // shift all the contents over.  a bit slow, but this is for debugging
        NSRange replRange = { static_cast<NSUInteger>(currPos), [result length] - currPos };
        [result replaceOccurrencesOfString:@"\n" withString:@"\n        " options:0 range:replRange];
        
        [result appendString:@"\n"];
    }

    [result appendString:@"\n--------------------------------------------\n"];    

    return result;
}

- (void)setPageCacheSize:(NSUInteger)size
{
    [core(self)->webView() setUsesPageCache:size != 0];
}

- (NSUInteger)pageCacheSize
{
    return [core(self)->webView() usesPageCache] ? WebCore::BackForwardCache::singleton().maxSize() : 0;
}

- (int)backListCount
{
    return core(self)->backListCount();
}

- (int)forwardListCount
{
    return core(self)->forwardListCount();
}

- (WebHistoryItem *)itemAtIndex:(int)index
{
    if (auto* mainFrame = core([core(self)->webView() mainFrame]))
        return retainPtr(kit(core(self)->itemAtIndex(index, mainFrame->frameID()).get())).autorelease();
    ASSERT_NOT_REACHED();
    return nullptr;
}

@end
