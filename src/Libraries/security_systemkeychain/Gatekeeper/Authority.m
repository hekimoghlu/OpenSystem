/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
//  Copyright (c) 2012 Apple. All rights reserved.
//

#import "Authority.h"

#import <Security/SecAssessment.h>

@implementation Authority


- (Authority *)initWithAssessment:(NSDictionary *)assessment
{
    [self updateWithAssessment:assessment];
    return self;
}

- (void)updateWithAssessment:(NSDictionary *)assessment
{
    self.identity = [assessment objectForKey:(__bridge id)kSecAssessmentRuleKeyID];
    self.remarks = [assessment objectForKey:(__bridge id)kSecAssessmentRuleKeyRemarks];
    self.disabled = [assessment objectForKey:(__bridge id)kSecAssessmentRuleKeyDisabled];
    self.codeRequirement = [assessment objectForKey:(__bridge id)kSecAssessmentRuleKeyRequirement];
    self.bookmark = [assessment objectForKey:(__bridge id)kSecAssessmentRuleKeyBookmark];
}

- (NSString *)description
{
    if (self.remarks)
	return self.remarks;
    return @"description here";
}

- (NSImage *)icon
{
    if (self.bookmark == NULL)
	return NULL;
    
    NSURL *url = [NSURL URLByResolvingBookmarkData:self.bookmark options:0 relativeToURL:NULL bookmarkDataIsStale:NULL error:NULL];
 
    NSDictionary *icons = [url resourceValuesForKeys:@[ NSURLEffectiveIconKey, NSURLCustomIconKey ] error:NULL];
    
    NSImage *image = [icons objectForKey: NSURLCustomIconKey];
    if (image)
	return image;

    return [icons objectForKey: NSURLEffectiveIconKey];
}

@end
