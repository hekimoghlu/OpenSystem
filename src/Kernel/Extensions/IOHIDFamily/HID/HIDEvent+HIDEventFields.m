/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
#import <HID/HIDEvent+HIDEventFields.h>

#import <HID/HIDEventFields_Internal.h>
#import <AssertMacros.h>

NS_ASSUME_NONNULL_BEGIN

#define IS_VALID_EVENT_FIELD(info) (info.field == 0 && info.fieldType == kEventFieldDataType_None && info.name == NULL) ? 0 : 1

@implementation HIDEvent (HIDEventDesc)

-(void) enumerateFieldsWithBlock:(HIDEventFieldInfoBlock) block
{
    HIDEventFieldInfo *info = nil;
    NSInteger index = 0;
    require(block, exit);
    
    info = [self getEventFields];
    require(info, exit);
    
    while(IS_VALID_EVENT_FIELD(info[index])) {
        block(&info[index]);
        index++;
    }
exit:
    return;
}
-(HIDEventFieldInfo* __nullable) getEventFields
{
    HIDEventFieldInfo *info = NULL;
    NSInteger index = 0;
    while (!info && hidEventFieldDescTable[index].type != kIOHIDEventTypeCount) {
        
        if (hidEventFieldDescTable[index].type != self.type) {
            index++;
            continue;
        }
        
        if (!hidEventFieldDescTable[index].selectors) {
            info = hidEventFieldDescTable[index].eventFieldDescTable;
        } else {
            
            NSInteger selectorIndex = 0;
            while (!info && hidEventFieldDescTable[index].selectors[selectorIndex].selectorTables != NULL) {
                
                
                NSInteger selectorValueIndex = 0;
                NSUInteger selectorValue =  (NSUInteger)[self integerValueForField:hidEventFieldDescTable[index].selectors[selectorIndex].value];
                
                while(!info && hidEventFieldDescTable[index].selectors[selectorIndex].selectorTables[selectorValueIndex].eventFieldDescTable
                      != NULL) {
                    
                    if (hidEventFieldDescTable[index].selectors[selectorIndex].selectorTables[selectorValueIndex].value != selectorValue) {
                        selectorValueIndex++;
                        continue;
                    }
                    
                    info = hidEventFieldDescTable[index].selectors[selectorIndex].selectorTables[selectorValueIndex].eventFieldDescTable;
                    selectorValueIndex++;
                }
                
                selectorIndex++;
            }
            
        }
        
        index++;
    }
    
    return info;
}

@end

NS_ASSUME_NONNULL_END


