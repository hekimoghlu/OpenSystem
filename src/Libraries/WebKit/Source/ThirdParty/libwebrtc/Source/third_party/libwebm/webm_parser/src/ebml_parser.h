/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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

// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_EBML_PARSER_H_
#define SRC_EBML_PARSER_H_

#include "src/byte_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec references:
// http://matroska.org/technical/specs/index.html#EBML
// https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown#ebml-header-elements
// http://www.webmproject.org/docs/container/#EBML
class EbmlParser : public MasterValueParser<Ebml> {
 public:
  EbmlParser()
      : MasterValueParser<Ebml>(
            MakeChild<UnsignedIntParser>(Id::kEbmlVersion, &Ebml::ebml_version),
            MakeChild<UnsignedIntParser>(Id::kEbmlReadVersion,
                                         &Ebml::ebml_read_version),
            MakeChild<UnsignedIntParser>(Id::kEbmlMaxIdLength,
                                         &Ebml::ebml_max_id_length),
            MakeChild<UnsignedIntParser>(Id::kEbmlMaxSizeLength,
                                         &Ebml::ebml_max_size_length),
            MakeChild<StringParser>(Id::kDocType, &Ebml::doc_type),
            MakeChild<UnsignedIntParser>(Id::kDocTypeVersion,
                                         &Ebml::doc_type_version),
            MakeChild<UnsignedIntParser>(Id::kDocTypeReadVersion,
                                         &Ebml::doc_type_read_version)) {}

 protected:
  Status OnParseCompleted(Callback* callback) override {
    return callback->OnEbml(metadata(Id::kEbml), value());
  }
};

}  // namespace webm

#endif  // SRC_EBML_PARSER_H_
