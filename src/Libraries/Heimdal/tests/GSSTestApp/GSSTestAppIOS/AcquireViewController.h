/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
//  Copyright (c) 2013 Apple, Inc. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "CredentialTableController.h"

@interface AcquireViewController : UIViewController <GSSCredentialsChangeNotification>

@property (weak) IBOutlet UITextField *username;
@property (weak) IBOutlet UISwitch *doPassword;
@property (weak) IBOutlet UITextField *password;
@property (weak) IBOutlet UISwitch *doCertificate;
@property (weak) IBOutlet UILabel *certificateLabel;
@property (weak) IBOutlet UILabel *statusLabel;
@property (weak) IBOutlet UITextField *kdchostname;


@property (weak) IBOutlet UITableView *credentialsTableView;
@property (assign) CredentialTableController *credentialsTableController;

@end
