/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
//  ViewController.h
//

#import <UIKit/UIKit.h>

@interface ViewController : UIViewController {
    dispatch_queue_t _queue;
}

@property (nonatomic, retain) IBOutlet  UITextView *ticketView;
@property (nonatomic, retain) IBOutlet  UITextField *authServerName;
@property (nonatomic, retain) IBOutlet  UITextField *authServerResult;
@property (nonatomic, retain) IBOutlet  UITextField *urlTextField;
@property (nonatomic, retain) IBOutlet  UITextView *urlResultTextView;

- (IBAction)addCredential:(id)sender;

@end
