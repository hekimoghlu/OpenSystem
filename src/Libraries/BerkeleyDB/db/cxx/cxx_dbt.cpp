/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
#include "db_config.h"

#include "db_int.h"

#include "db_cxx.h"
#include "dbinc/cxx_int.h"

#include "dbinc/db_page.h"
#include "dbinc_auto/db_auto.h"
#include "dbinc_auto/crdel_auto.h"
#include "dbinc/db_dispatch.h"
#include "dbinc_auto/db_ext.h"
#include "dbinc_auto/common_ext.h"

Dbt::Dbt()
{
	DBT *dbt = this;
	memset(dbt, 0, sizeof(DBT));
}

Dbt::Dbt(void *data_arg, u_int32_t size_arg)
{
	DBT *dbt = this;
	memset(dbt, 0, sizeof(DBT));
	set_data(data_arg);
	set_size(size_arg);
}

Dbt::~Dbt()
{
}

Dbt::Dbt(const Dbt &that)
{
	const DBT *from = &that;
	DBT *to = this;
	memcpy(to, from, sizeof(DBT));
}

Dbt &Dbt::operator = (const Dbt &that)
{
	if (this != &that) {
		const DBT *from = &that;
		DBT *to = this;
		memcpy(to, from, sizeof(DBT));
	}
	return (*this);
}
