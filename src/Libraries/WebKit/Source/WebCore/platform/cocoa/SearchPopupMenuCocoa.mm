/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
#import "SearchPopupMenuCocoa.h"

#import <wtf/cocoa/VectorCocoa.h>

namespace WebCore {

static NSString * const dateKey = @"date";
static NSString * const itemsKey = @"items";
static NSString * const searchesKey = @"searches";
static NSString * const searchStringKey = @"searchString";

static NSString *searchFieldRecentSearchesPlistPath(NSString *baseDirectory)
{
    return [baseDirectory stringByAppendingPathComponent:@"RecentSearches.plist"];
}

static RetainPtr<NSMutableDictionary> readSearchFieldRecentSearchesPlist(NSString *baseDirectory)
{
    return adoptNS([[NSMutableDictionary alloc] initWithContentsOfFile:searchFieldRecentSearchesPlistPath(baseDirectory)]);
}

static WallTime toSystemClockTime(NSDate *date)
{
    ASSERT(date);
    return WallTime::fromRawSeconds(date.timeIntervalSince1970);
}

static NSDate *toNSDateFromSystemClock(WallTime time)
{
    return [NSDate dateWithTimeIntervalSince1970:time.secondsSinceEpoch().seconds()];
}

static NSMutableArray *typeCheckedRecentSearchesArray(NSMutableDictionary *itemsDictionary, NSString *name)
{
    NSMutableDictionary *nameDictionary = [itemsDictionary objectForKey:name];
    if (![nameDictionary isKindOfClass:[NSDictionary class]])
        return nil;

    NSMutableArray *recentSearches = [nameDictionary objectForKey:searchesKey];
    if (![recentSearches isKindOfClass:[NSArray class]])
        return nil;

    return recentSearches;
}

static NSDate *typeCheckedDateInRecentSearch(NSDictionary *recentSearch)
{
    if (![recentSearch isKindOfClass:[NSDictionary class]])
        return nil;

    NSDate *date = [recentSearch objectForKey:dateKey];
    if (![date isKindOfClass:[NSDate class]])
        return nil;

    return date;
}

static RetainPtr<NSDictionary> typeCheckedRecentSearchesRemovingRecentSearchesAddedAfterDate(NSDate *date, const String& directory)
{
    if ([date isEqualToDate:[NSDate distantPast]])
        return nil;

    RetainPtr<NSMutableDictionary> recentSearchesPlist = readSearchFieldRecentSearchesPlist(directory);
    NSMutableDictionary *itemsDictionary = [recentSearchesPlist objectForKey:itemsKey];
    if (![itemsDictionary isKindOfClass:[NSDictionary class]])
        return nil;

    RetainPtr<NSMutableArray> keysToRemove = adoptNS([[NSMutableArray alloc] init]);
    for (NSString *key in itemsDictionary) {
        if (![key isKindOfClass:[NSString class]])
            return nil;

        NSMutableArray *recentSearches = typeCheckedRecentSearchesArray(itemsDictionary, key);
        if (!recentSearches)
            return nil;

        RetainPtr<NSMutableArray> entriesToRemove = adoptNS([[NSMutableArray alloc] init]);
        for (NSDictionary *recentSearch in recentSearches) {
            NSDate *dateAdded = typeCheckedDateInRecentSearch(recentSearch);
            if (!dateAdded)
                return nil;

            if ([dateAdded compare:date] == NSOrderedDescending)
                [entriesToRemove addObject:recentSearch];
        }

        [recentSearches removeObjectsInArray:entriesToRemove.get()];
        if (!recentSearches.count)
            [keysToRemove addObject:key];
    }

    [itemsDictionary removeObjectsForKeys:keysToRemove.get()];
    if (!itemsDictionary.count)
        return nil;

    return recentSearchesPlist;
}

void saveRecentSearchesToFile(const String& name, const Vector<RecentSearch>& searchItems, const String& directory)
{
    if (name.isEmpty() || directory.isEmpty())
        return;

    RetainPtr<NSDictionary> recentSearchesPlist = readSearchFieldRecentSearchesPlist(directory);
    RetainPtr<NSMutableDictionary> itemsDictionary = [recentSearchesPlist objectForKey:itemsKey];

    // The NSMutableDictionary method we use to read the property list guarantees we get only
    // mutable containers, but it does not guarantee the file has a dictionary as expected.
    if (![itemsDictionary isKindOfClass:[NSDictionary class]]) {
        itemsDictionary = adoptNS([[NSMutableDictionary alloc] init]);
        recentSearchesPlist = adoptNS([[NSDictionary alloc] initWithObjectsAndKeys:itemsDictionary.get(), itemsKey, nil]);
    }

    if (searchItems.isEmpty())
        [itemsDictionary removeObjectForKey:name];
    else {
        auto items = createNSArray(searchItems, [] (auto& item) {
            return adoptNS([[NSDictionary alloc] initWithObjectsAndKeys:item.string, searchStringKey, toNSDateFromSystemClock(item.time), dateKey, nil]);
        });
        [itemsDictionary setObject:adoptNS([[NSDictionary alloc] initWithObjectsAndKeys:items.get(), searchesKey, nil]).get() forKey:name];
    }

    [recentSearchesPlist writeToFile:searchFieldRecentSearchesPlistPath(directory) atomically:YES];
}

Vector<RecentSearch> loadRecentSearchesFromFile(const String& name, const String& directory)
{
    Vector<RecentSearch> searchItems;
    if (name.isEmpty() || directory.isEmpty())
        return searchItems;

    RetainPtr<NSMutableDictionary> recentSearchesPlist = readSearchFieldRecentSearchesPlist(directory);
    if (!recentSearchesPlist)
        return searchItems;

    NSMutableDictionary *items = [recentSearchesPlist objectForKey:itemsKey];
    if (![items isKindOfClass:[NSDictionary class]])
        return searchItems;

    NSArray *recentSearches = typeCheckedRecentSearchesArray(items, name);
    if (!recentSearches)
        return searchItems;
    
    for (NSDictionary *item in recentSearches) {
        NSDate *date = typeCheckedDateInRecentSearch(item);
        if (!date)
            continue;

        NSString *searchString = [item objectForKey:searchStringKey];
        if (![searchString isKindOfClass:[NSString class]])
            continue;
        
        searchItems.append({ String{ searchString }, toSystemClockTime(date) });
    }

    return searchItems;
}

void removeRecentlyModifiedRecentSearchesFromFile(WallTime oldestTimeToRemove, const String& directory)
{
    if (directory.isEmpty())
        return;

    NSString *plistPath = searchFieldRecentSearchesPlistPath(directory);
    NSDate *date = toNSDateFromSystemClock(oldestTimeToRemove);
    auto recentSearchesPlist = typeCheckedRecentSearchesRemovingRecentSearchesAddedAfterDate(date, directory);
    if (recentSearchesPlist)
        [recentSearchesPlist writeToFile:plistPath atomically:YES];
    else {
        auto emptyItemsDictionary = adoptNS([[NSDictionary alloc] init]);
        auto emptyRecentSearchesDictionary = adoptNS([[NSDictionary alloc] initWithObjectsAndKeys:emptyItemsDictionary.get(), itemsKey, nil]);
        [emptyRecentSearchesDictionary writeToFile:plistPath atomically:YES];
    }
}

}
