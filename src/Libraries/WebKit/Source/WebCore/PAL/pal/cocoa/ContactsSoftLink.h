/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
#pragma once

#if HAVE(CONTACTS)

#import <Contacts/Contacts.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(PAL, Contacts)
SOFT_LINK_CLASS_FOR_HEADER(PAL, CNContact)
SOFT_LINK_CLASS_FOR_HEADER(PAL, CNLabeledValue)
SOFT_LINK_CLASS_FOR_HEADER(PAL, CNPhoneNumber)
SOFT_LINK_CLASS_FOR_HEADER(PAL, CNPostalAddress)
SOFT_LINK_CLASS_FOR_HEADER(PAL, CNMutableContact)
SOFT_LINK_CLASS_FOR_HEADER(PAL, CNMutablePostalAddress)
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactDepartmentNameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactFamilyNameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactGivenNameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactJobTitleKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactMiddleNameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactNamePrefixKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactNameSuffixKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactNicknameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactNoteKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactOrganizationNameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactPhoneticFamilyNameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactPhoneticGivenNameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactPhoneticMiddleNameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactPhoneticOrganizationNameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactPreviousFamilyNameKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactBirthdayKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactNonGregorianBirthdayKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactPhoneNumbersKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactEmailAddressesKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactPostalAddressesKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactDatesKey, NSString *);
SOFT_LINK_CONSTANT_FOR_HEADER(PAL, Contacts, CNContactUrlAddressesKey, NSString *);

#endif
