/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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
// *****************************************************************************
//  cclparser.h
//
//  Created by kevine on 3/1/06.
//  Recreated by sspies 15 January 2007.
// *****************************************************************************

#import <Foundation/Foundation.h>
#import "cclkeys.h"

#define kCCLOtherVendorName @"Other"

@interface CCLParser : NSObject 
{
    NSMutableDictionary *mBundleData;       // vendors->models->personalities
    NSMutableDictionary *mBundlesProcessed; // bundle IDs -> paths
    NSMutableDictionary *mFlatOverrides;    // flat names -> personalities
                                            // (contains "Supersedes" data)

    NSSet   *mTypeFilter;
}

// allocate a parser
+ (CCLParser*)createCCLParser;

// sets a filter for CCL bundle-based personalities.  Specifically, 
// desiredConnectTypes causes the -process* routines to silently ignore
// personalities with Connect Type values other than those listed.
// (e.g. "GPRS" or "Dialup"; someday WWAN?)
// Flat CCL bundles (which have no type information) are still included.
// Be sure to call -clearParser and then the process* routines again
// if changing the filter list for an existing object.
- (void)setTypeFilter:(NSSet*)desiredConnectTypes;

// recursively searches directory (i.e. /Library/Modem Scripts) for CCLs.
// Additional invocations add to the store.
//
// A CCL bundle is a directory with the .ccl extension.
// processFolder: returns NO if it finds any directory that looks like a
// CCL bundle but isn't a properly formed (doesn't have the right files, bad
// version, etc).  It does not give up just because it found one malformed
// CCL bundle.  Conforming bundles still have their personality data added.  
// Any files are assumed to be flat CCL scripts and are gathered together
// under the 'Other' vendor.  "Flat" CCL personalities have their DeviceVendor
// property set to the English string "Other" (kCCLOtherVendorName).
// Their model is the CCL filename.
- (BOOL)processFolder:(NSString*)folderPath;

// add a single bundle or flat CCL script to the store
- (BOOL)processCCLBundle:(NSString*)path;
- (BOOL)processFlatCCL:(NSString*)path named:(NSString*)name;

// expands names to remove ambiguity; will leave expanded dups.
// Call after adding all CCLs.
- (void)cleanupDuplicates;

// returns a new array of vendor keys sorted alphabetically except for
// 'Other' which will be appended to the list (if there were flat CCLs)
// 'Other' should appear in a separate segment of the Vendor popup and
// be localized by the callers of copyVendorList.  Additionally, OS X
// contains a number of bundles with DeviceVendor = "Generic".  Generic
// should also be localized.
- (NSArray*)copyVendorList;

// returns a reference to a sorted (by model name) list of personalities for
// one of the 'copyVendorList' vendors.  dictionary keys from cclkeys.h.
- (NSArray*)getModelListForVendor:(NSString*)vendor;

// attempts to upgrade a pre-Leopard deviceConfiguration dictionary to have a 
// vendor, model, connection script, and personality if needed.
// Only needed if vendor/model missing from stored device configuration.
// If vendor/model are present, they are validated and the ConnectionScript
// updated or nil returned if there was no match.  Beware -setTypeFilter:.
- (NSMutableDictionary*)upgradeDeviceConfiguration:(NSDictionary*)deviceConf;

// merges personality data (e.g. preferred APN/CID) with provided
// SystemConfiguration device (e.g. modem) configuration dictionary.
// returns autoreleased NSMutableDictionary on success; NULL on failure.
// (The extra copy makes sure we only store what the user chooses or types.
// i.e. One personality's defaults don't end up in the wrong personality.)
- (NSMutableDictionary*)mergeCCLPersonality:(NSDictionary*)personality withDeviceConfiguration:(NSDictionary*)deviceConfiguration;


// empties the store, retaining any type filters
- (void)clearParser;

@end
