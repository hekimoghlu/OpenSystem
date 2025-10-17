/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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

#include "regint.h"

/*
 * Name: UTF8-DoCoMo, SJIS-DoCoMo
 * Link: https://www.nttdocomo.co.jp/english/service/developer/make/content/pictograph/basic/index.html
 * Link: https://www.nttdocomo.co.jp/english/service/developer/make/content/pictograph/extention/index.html
 */
ENC_REPLICATE("UTF8-DoCoMo", "UTF-8")
ENC_REPLICATE("SJIS-DoCoMo", "Windows-31J")

/*
 * Name: UTF8-KDDI, SJIS-KDDI, ISO-2022-JP-KDDI
 * Link: http://www.au.kddi.com/ezfactory/tec/spec/img/typeD.pdf
 */
ENC_REPLICATE("UTF8-KDDI", "UTF-8")
ENC_REPLICATE("SJIS-KDDI", "Windows-31J")
ENC_REPLICATE("ISO-2022-JP-KDDI", "ISO-2022-JP")
ENC_REPLICATE("stateless-ISO-2022-JP-KDDI", "stateless-ISO-2022-JP")

/*
 * Name: UTF8-SoftBank, SJIS-SoftBank
 * Link: http://creation.mb.softbank.jp/web/web_pic_about.html
 * Link: http://www2.developers.softbankmobile.co.jp/dp/tool_dl/download.php?docid=120&companyid=
 */
ENC_REPLICATE("UTF8-SoftBank", "UTF-8")
ENC_REPLICATE("SJIS-SoftBank", "Windows-31J")
