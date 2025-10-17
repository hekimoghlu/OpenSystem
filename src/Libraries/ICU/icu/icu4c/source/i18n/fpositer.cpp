/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
******************************************************************************
* Copyright (C) 2009-2012, International Business Machines Corporation and
* others. All Rights Reserved.
******************************************************************************
*   Date        Name        Description
*   12/14/09    doug        Creation.
******************************************************************************
*/

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/fpositer.h"
#include "cmemory.h"
#include "uvectr32.h"

U_NAMESPACE_BEGIN

FieldPositionIterator::~FieldPositionIterator() {
  delete data;
  data = nullptr;
  pos = -1;
}

FieldPositionIterator::FieldPositionIterator()
    : data(nullptr), pos(-1) {
}

FieldPositionIterator::FieldPositionIterator(const FieldPositionIterator &rhs)
  : UObject(rhs), data(nullptr), pos(rhs.pos) {

  if (rhs.data) {
    UErrorCode status = U_ZERO_ERROR;
    data = new UVector32(status);
    data->assign(*rhs.data, status);
    if (status != U_ZERO_ERROR) {
      delete data;
      data = nullptr;
      pos = -1;
    }
  }
}

bool FieldPositionIterator::operator==(const FieldPositionIterator &rhs) const {
  if (&rhs == this) {
    return true;
  }
  if (pos != rhs.pos) {
    return false;
  }
  if (!data) {
    return rhs.data == nullptr;
  }
  return rhs.data ? data->operator==(*rhs.data) : false;
}

void FieldPositionIterator::setData(UVector32 *adopt, UErrorCode& status) {
  // Verify that adopt has valid data, and update status if it doesn't.
  if (U_SUCCESS(status)) {
    if (adopt) {
      if (adopt->size() == 0) {
        delete adopt;
        adopt = nullptr;
      } else if ((adopt->size() % 4) != 0) {
        status = U_ILLEGAL_ARGUMENT_ERROR;
      } else {
        for (int i = 2; i < adopt->size(); i += 4) {
          if (adopt->elementAti(i) >= adopt->elementAti(i+1)) {
            status = U_ILLEGAL_ARGUMENT_ERROR;
            break;
          }
        }
      }
    }
  }

  // We own the data, even if status is in error, so we need to delete it now
  // if we're not keeping track of it.
  if (!U_SUCCESS(status)) {
    delete adopt;
    return;
  }

  delete data;
  data = adopt;
  pos = adopt == nullptr ? -1 : 0;
}

UBool FieldPositionIterator::next(FieldPosition& fp) {
  if (pos == -1) {
    return false;
  }

  // Ignore the first element of the tetrad: used for field category
  pos++;
  fp.setField(data->elementAti(pos++));
  fp.setBeginIndex(data->elementAti(pos++));
  fp.setEndIndex(data->elementAti(pos++));

  if (pos == data->size()) {
    pos = -1;
  }

  return true;
}

U_NAMESPACE_END

#endif /* #if !UCONFIG_NO_FORMATTING */

