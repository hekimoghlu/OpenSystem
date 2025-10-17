/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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

DbMultipleIterator::DbMultipleIterator(const Dbt &dbt)
 : data_((u_int8_t*)dbt.get_data()),
   p_((u_int32_t*)(data_ + dbt.get_ulen() - sizeof(u_int32_t)))
{
}

bool DbMultipleDataIterator::next(Dbt &data)
{
	if (*p_ == (u_int32_t)-1) {
		data.set_data(0);
		data.set_size(0);
		p_ = 0;
	} else {
		data.set_data(data_ + *p_--);
		data.set_size(*p_--);
		if (data.get_size() == 0 && data.get_data() == data_)
			data.set_data(0);
	}
	return (p_ != 0);
}

bool DbMultipleKeyDataIterator::next(Dbt &key, Dbt &data)
{
	if (*p_ == (u_int32_t)-1) {
		key.set_data(0);
		key.set_size(0);
		data.set_data(0);
		data.set_size(0);
		p_ = 0;
	} else {
		key.set_data(data_ + *p_--);
		key.set_size(*p_--);
		data.set_data(data_ + *p_--);
		data.set_size(*p_--);
	}
	return (p_ != 0);
}

bool DbMultipleRecnoDataIterator::next(db_recno_t &recno, Dbt &data)
{
	if (*p_ == (u_int32_t)0) {
		recno = 0;
		data.set_data(0);
		data.set_size(0);
		p_ = 0;
	} else {
		recno = *p_--;
		data.set_data(data_ + *p_--);
		data.set_size(*p_--);
	}
	return (p_ != 0);
}
