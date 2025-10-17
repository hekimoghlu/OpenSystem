/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#if ENABLE(WK_WEB_EXTENSIONS)

#import <WebCore/Icon.h>
#import <wtf/HashSet.h>
#import <wtf/OptionSet.h>
#import <wtf/RetainPtr.h>
#import <wtf/URLHash.h>
#import <wtf/UUID.h>
#import <wtf/WallTime.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/text/StringHash.h>

OBJC_CLASS NSArray;
OBJC_CLASS NSDate;
OBJC_CLASS NSDictionary;
OBJC_CLASS NSError;
OBJC_CLASS NSLocale;
OBJC_CLASS NSSet;
OBJC_CLASS NSString;
OBJC_CLASS NSUUID;

#define THROW_UNLESS(condition, message) \
    if (UNLIKELY(!(condition))) \
        [NSException raise:NSInternalInconsistencyException format:message]

namespace API {
class Data;
}

namespace WebKit {

template<typename T> T *filterObjects(T *container, bool NS_NOESCAPE (^block)(id key, id value));
template<> NSArray *filterObjects<NSArray>(NSArray *, bool NS_NOESCAPE (^block)(id key, id value));
template<> NSDictionary *filterObjects<NSDictionary>(NSDictionary *, bool NS_NOESCAPE (^block)(id key, id value));
template<> NSSet *filterObjects<NSSet>(NSSet *, bool NS_NOESCAPE (^block)(id key, id value));

template<typename T>
T *filterObjects(const RetainPtr<T>& container, bool NS_NOESCAPE (^block)(id key, id value))
{
    return filterObjects<T>(container.get(), block);
}

template<typename T> T *mapObjects(T *container, id NS_NOESCAPE (^block)(id key, id value));
template<> NSArray *mapObjects<NSArray>(NSArray *, id NS_NOESCAPE (^block)(id key, id value));
template<> NSDictionary *mapObjects<NSDictionary>(NSDictionary *, id NS_NOESCAPE (^block)(id key, id value));
template<> NSSet *mapObjects<NSSet>(NSSet *, id NS_NOESCAPE (^block)(id key, id value));

template<typename T>
T *mapObjects(const RetainPtr<T>& container, id NS_NOESCAPE (^block)(id key, id value))
{
    return mapObjects<T>(container.get(), block);
}

template<typename T>
T *objectForKey(NSDictionary *dictionary, id key, bool returningNilIfEmpty = true, Class containingObjectsOfClass = Nil)
{
    ASSERT(returningNilIfEmpty);
    ASSERT(!containingObjectsOfClass);
    // Specialized implementations in CocoaHelpers.mm handle returningNilIfEmpty and containingObjectsOfClass for different Foundation types.
    return dynamic_objc_cast<T>(dictionary[key]);
}

template<typename T>
T *objectForKey(const RetainPtr<NSDictionary>& dictionary, id key, bool returningNilIfEmpty = true, Class containingObjectsOfClass = Nil)
{
    return objectForKey<T>(dictionary.get(), key, returningNilIfEmpty, containingObjectsOfClass);
}

inline bool boolForKey(NSDictionary *dictionary, id key, bool defaultValue)
{
    NSNumber *value = dynamic_objc_cast<NSNumber>(dictionary[key]);
    return value ? value.boolValue : defaultValue;
}

template<typename T>
inline std::optional<RetainPtr<T>> toOptional(T *maybeNil)
{
    if (maybeNil)
        return maybeNil;
    return std::nullopt;
}

inline std::optional<String> toOptional(NSString *maybeNil)
{
    if (maybeNil)
        return maybeNil;
    return std::nullopt;
}

inline CocoaImage *toCocoaImage(RefPtr<WebCore::Icon> icon)
{
    return icon ? icon->image().get() : nil;
}

enum class JSONOptions {
    FragmentsAllowed = 1 << 0, /// Allows for top-level scalar types, in addition to arrays and dictionaries.
};

using JSONOptionSet = OptionSet<JSONOptions>;

bool isValidJSONObject(id, JSONOptionSet = { });

id parseJSON(NSString *, JSONOptionSet = { }, NSError ** = nullptr);
id parseJSON(NSData *, JSONOptionSet = { }, NSError ** = nullptr);
id parseJSON(API::Data&, JSONOptionSet = { }, NSError ** = nullptr);

NSString *encodeJSONString(id, JSONOptionSet = { }, NSError ** = nullptr);
NSData *encodeJSONData(id, JSONOptionSet = { }, NSError ** = nullptr);

NSDictionary *dictionaryWithLowercaseKeys(NSDictionary *);
NSDictionary *dictionaryWithKeys(NSDictionary *, NSArray *keys);
NSDictionary *mergeDictionaries(NSDictionary *, NSDictionary *);
NSDictionary *mergeDictionariesAndSetValues(NSDictionary *, NSDictionary *);

NSString *privacyPreservingDescription(NSError *);

NSURL *ensureDirectoryExists(NSURL *directory);

NSString *escapeCharactersInString(NSString *, NSString *charactersToEscape);

void callAfterRandomDelay(Function<void()>&&);

NSDate *toAPI(const WallTime&);
WallTime toImpl(NSDate *);

NSSet *toAPI(const HashSet<URL>&);

NSSet *toAPI(const HashSet<String>&);
NSArray *toAPIArray(const HashSet<String>&);
HashSet<String> toImpl(NSSet *);

using DataMap = HashMap<String, std::variant<String, Ref<API::Data>>>;
DataMap toDataMap(NSDictionary *);

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
