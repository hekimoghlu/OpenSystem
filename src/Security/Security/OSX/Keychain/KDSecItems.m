/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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
#import "KDSecItems.h"
#include <Security/Security.h>
#include <Security/SecItemPriv.h>


NSString *kKDSecItemsUpdated = @"KDSecItemsUpdated";

@interface KDSecItems ()
@property NSMutableArray *items;
@end

@implementation KDSecItems

-(NSInteger)numberOfRowsInTableView:(NSTableView*)t
{
    return [self.items count];
}

+(NSString*)nameOfItem:(NSDictionary*)item
{
    id name = item[(id)kSecAttrService];
    if (name) {
        return name;
    }
    
    NSString *path = item[(id)kSecAttrPath];
    if (!path) {
        path = @"/";
    }
    NSString *port = item[(id)kSecAttrPort];
    if ([@"0" isEqualToString:port] || [@0 isEqual:port]) {
        port = @"";
    } else {
        port = [NSString stringWithFormat:@":%@", port];
    }
    
    return [NSString stringWithFormat:@"%@://%@%@%@", item[(id)kSecAttrProtocol], item[(id)kSecAttrServer], port, path];
}

- (id)tableView:(NSTableView *)aTableView objectValueForTableColumn:(NSTableColumn *)aTableColumn row:(NSInteger)rowIndex
{
    NSString *identifier = [aTableColumn identifier];
    
    if ([@"account" isEqualToString:identifier]) {
        return self.items[rowIndex][(id)kSecAttrAccount];
    }
    if ([@"name" isEqualToString:identifier]) {
        return [KDSecItems nameOfItem:self.items[rowIndex]];
    }
    
    return [NSString stringWithFormat:@"*** c=%@ r%ld", [aTableColumn identifier], (long)rowIndex];
}

-(NSArray*)fetchItemsMatching:(NSDictionary *)query
{
    CFTypeRef raw_items = NULL;
    OSStatus result = SecItemCopyMatching((__bridge CFDictionaryRef)(query), &raw_items);
    if (result) {
        // XXX: UI
        NSLog(@"Error result %d - query: %@", result, query);
        return nil;
    }
    if (CFArrayGetTypeID() == CFGetTypeID(raw_items)) {
        return (__bridge NSArray*)raw_items;
    }
    
    NSLog(@"Unexpected result type from copyMatching: %@ (query=%@)", raw_items, query);
    CFRelease(raw_items);

    return nil;
}

-(void)loadItems
{
    NSDictionary *query_genp = @{(id)kSecClass: (id)kSecClassGenericPassword,
                                 (__bridge id)kSecAttrSynchronizable: @1,
                                 (id)kSecMatchLimit: (id)kSecMatchLimitAll,
                                 (id)kSecReturnAttributes: (id)kCFBooleanTrue};
    NSDictionary *query_inet = @{(id)kSecClass: (id)kSecClassInternetPassword,
                                 (__bridge id)kSecAttrSynchronizable: @1,
                                 (id)kSecMatchLimit: (id)kSecMatchLimitAll,
                                 (id)kSecReturnAttributes: (id)kCFBooleanTrue};
    NSArray *nextItems = [[self fetchItemsMatching:query_genp] arrayByAddingObjectsFromArray:[self fetchItemsMatching:query_inet]];
    self.items = [[nextItems sortedArrayUsingComparator:^NSComparisonResult(id a, id b) {
        NSDictionary *da = a, *db = b;
        return [da[(id)kSecAttrService] caseInsensitiveCompare:db[(id)kSecAttrService]];
    }] mutableCopy];
        
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter] postNotificationName:kKDSecItemsUpdated object:self];
    });
}

-(id)init
{
    [self loadItems];
    return self;
}

@end
