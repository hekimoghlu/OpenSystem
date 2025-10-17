/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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

//
//  mark.m
//  Security
//
//  Created by Ben Williamson on 6/2/17.
//
//

#import "mark.h"
#import "Keychain.h"

NSDictionary<NSString *, NSString *> *markForIdent(NSString *ident)
{
    return @{
             [NSString stringWithFormat:@"mark-%@-a", ident]: @"I",
             [NSString stringWithFormat:@"mark-%@-b", ident]: @"am",
             [NSString stringWithFormat:@"mark-%@-c", ident]: @"done.",
             };
}

NSDictionary<NSString *, NSString *> *updateForIdent(NSString *ident)
{
    return @{
             [NSString stringWithFormat:@"mark-%@-a", ident]: @"Updated,",
             [NSString stringWithFormat:@"mark-%@-b", ident]: @"I",
             [NSString stringWithFormat:@"mark-%@-c", ident]: @"am.",
             };
}


void writeMark(NSString *ident, NSString *view)
{
    Keychain *keychain = [[Keychain alloc] init];
    NSDictionary<NSString *, NSString *> *dict = markForIdent(ident);
    [dict enumerateKeysAndObjectsUsingBlock:^(NSString *name, NSString *value, BOOL *stop) {
        NSLog(@"Writing mark in %@: %@ = %@", view, name, value);
        OSStatus status = [keychain addItem:name value:value view:view];
        switch (status) {
            case errSecSuccess:
                break;
            case errSecDuplicateItem:
                NSLog(@"(mark was already there, fine)");
                break;
            default:
                NSLog(@"Error writing mark %@: %d", name, (int)status);
                exit(1);
        }
    }];
}

void deleteMark(NSString *ident)
{
    Keychain *keychain = [[Keychain alloc] init];
    NSDictionary<NSString *, NSString *> *dict = markForIdent(ident);
    [dict enumerateKeysAndObjectsUsingBlock:^(NSString *name, NSString *value, BOOL *stop) {
        NSLog(@"Deleting mark: %@", name);
        [keychain deleteItemWithName:name];
    }];
}

void updateMark(NSString *ident)
{
    Keychain *keychain = [[Keychain alloc] init];
    NSDictionary<NSString *, NSString *> *dict = updateForIdent(ident);
    [dict enumerateKeysAndObjectsUsingBlock:^(NSString *name, NSString *value, BOOL *stop) {
        NSLog(@"Updating mark: %@ = %@", name, value);
        OSStatus status = [keychain updateItemWithName:name newValue:value];
        switch (status) {
            case errSecSuccess:
                break;
            case errSecDuplicateItem:
                NSLog(@"(updated mark was already there, fine)");
                break;
            default:
                NSLog(@"Error updating mark %@: %d", name, (int)status);
                exit(1);
        }
    }];
}

static BOOL verifyMarksGeneric(NSArray<NSString *> *idents, BOOL updated)
{
    Keychain *keychain = [[Keychain alloc] init];
    
    NSMutableDictionary<NSString *, NSString *> *expected = [NSMutableDictionary dictionary];
    for (NSString *ident in idents) {
        NSDictionary<NSString *, NSString *> *mark = nil;
        if (!updated) {
            mark = markForIdent(ident);
        } else {
            mark = updateForIdent(ident);
        }
        [expected addEntriesFromDictionary:mark];
    }
    
    NSDictionary<NSString *, NSArray *> *actual = [keychain getAllItems];
    NSMutableDictionary<NSString *, NSString *> *actualNoPRefs = [NSMutableDictionary dictionary];
    for (NSString *name in actual) {
        actualNoPRefs[name] = actual[name][1];
    }
    NSLog(@"verifyMarks - getAllItems got %lu items", (unsigned long)actual.count);
    
    if ([actualNoPRefs isEqualToDictionary:expected]) {
        NSLog(@"Verify passed");
        return YES;
    }
    NSLog(@"Verify failed for idents %@", idents);
    
    for (NSString *name in actual) {
        if (!expected[name]) {
            NSLog(@"- unexpected item %@ %@ = %@", actual[name][0], name, actual[name][1]);
        }
    }
    for (NSString *name in expected) {
        if (!actual[name]) {
            NSLog(@"- missing item %@", name);
            continue;
        }
        if (![actual[name][1] isEqualToString:expected[name]]) {
            NSLog(@"- incorrect item %@ %@ = %@ should be %@",  actual[name][0], name, actual[name][1], expected[name]);
        }
    }
    return NO;
}

BOOL verifyMarks(NSArray<NSString *> *idents)
{
    return verifyMarksGeneric(idents, NO);
}

BOOL verifyUpdateMarks(NSArray<NSString *> *idents)
{
    return verifyMarksGeneric(idents, YES);
}
