/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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

#include "db_cxx.h"
#include "stdlib.h"

void test1()
{
	int numberOfKeysToWrite= 10000;
	Db db(0,DB_CXX_NO_EXCEPTIONS);
	db.set_pagesize(512);
	int err= db.open(0, "test1.db", 0, DB_BTREE, DB_CREATE, 0);
	{
		int i= 0;
		Dbt key(&i,sizeof(i));
		Dbt data(&i,sizeof(i));
		for(;i<numberOfKeysToWrite;++i)
		{
			db.put(0,&key,&data,0);
		}
	}

	{
		Dbc *dbc;
		err= db.cursor(0,&dbc,0);

		char *check= (char*)calloc(numberOfKeysToWrite,1);
		char buffer[8192];
		int numberOfKeysRead= 0;
		Dbt multikey(&numberOfKeysRead,sizeof(numberOfKeysRead));
		Dbt multidata(&buffer,sizeof(buffer));
		multidata.set_flags(DB_DBT_USERMEM);
		multidata.set_ulen(sizeof(buffer));
		err= 0;
		while(err==0)
		{
			err= dbc->get(&multikey,&multidata,DB_NEXT|DB_MULTIPLE_KEY);
			if(err==0)
			{
				Dbt key, data;
				DbMultipleKeyDataIterator i(multidata);
				while(err==0 && i.next(key,data))
				{
					int actualKey= *((int*)key.get_data());
					int actualData= *((int*)data.get_data());
					if(actualKey!=actualData)
					{
						std::cout << "Error: key/data mismatch. " << actualKey << "!=" << actualData << std::endl;
						err= -1;
					}
					else
					{
						check[actualKey]++;
					}
					numberOfKeysRead++;
				}
			} else if(err!=DB_NOTFOUND)
				std::cout << "Error: dbc->get: " << db_strerror(err) << std::endl;
		}
		if(numberOfKeysRead!=numberOfKeysToWrite)
		{
			std::cout << "Error: key count mismatch. " << numberOfKeysRead << "!=" << numberOfKeysToWrite << std::endl;
		}
		for(int n=0;n<numberOfKeysToWrite;++n)
		{
			if(check[n]!=1)
			{
				std::cout << "Error: key " << n << " was written to the database, but not read back." << std::endl;
			}
		}
		free(check);
		dbc->close();
	}

	db.close(0);
}

void test2()
{
	int numberOfKeysToWrite= 10000;
	Db db(0,DB_CXX_NO_EXCEPTIONS);
	db.set_flags(DB_DUP);
	db.set_pagesize(512);
	int err= db.open(0, "test2.db", 0, DB_BTREE, DB_CREATE, 0);

	{
		int i= 0;
		int k= 0;
		Dbt key(&k,sizeof(k));
		Dbt data(&i,sizeof(i));
		for(;i<numberOfKeysToWrite;++i)
		{
			err= db.put(0,&key,&data,0);
		}
	}

	{
		Dbc *dbc;
		err= db.cursor(0,&dbc,0);

		char buffer[8192];
		int numberOfKeysRead= 0;
		Dbt multikey(&numberOfKeysRead,sizeof(numberOfKeysRead));
		Dbt multidata(&buffer,sizeof(buffer));
		multidata.set_flags(DB_DBT_USERMEM);
		multidata.set_ulen(sizeof(buffer));
		err= 0;
		while(err==0)
		{
			err= dbc->get(&multikey,&multidata,DB_NEXT|DB_MULTIPLE);
			if(err==0)
			{
				Dbt data;
				DbMultipleDataIterator i(multidata);
				while(err==0 && i.next(data))
				{
					int actualData= *((int*)data.get_data());
					if(numberOfKeysRead!=actualData)
					{
						std::cout << "Error: key/data mismatch. " << numberOfKeysRead << "!=" << actualData << std::endl;
						err= -1;
					}
					numberOfKeysRead++;
				}
			} else if(err!=DB_NOTFOUND)
				std::cout << "Error: dbc->get: " << db_strerror(err) << std::endl;
		}
		if(numberOfKeysRead!=numberOfKeysToWrite)
		{
			std::cout << "Error: key count mismatch. " << numberOfKeysRead << "!=" << numberOfKeysToWrite << std::endl;
		}
		dbc->close();
	}
	db.close(0);
}

void test3()
{
	int numberOfKeysToWrite= 10000;
	Db db(0,DB_CXX_NO_EXCEPTIONS);
	db.set_pagesize(512);
	int err= db.open(0, "test3.db", 0, DB_RECNO, DB_CREATE, 0);

	{
		int i= 0;
		Dbt key;
		Dbt data(&i,sizeof(i));
		for(;i<numberOfKeysToWrite;++i)
		{
			err= db.put(0,&key,&data,DB_APPEND);
		}
	}

	{
		Dbc *dbc;
		err= db.cursor(0,&dbc,0);

		char buffer[8192];
		int numberOfKeysRead= 0;
		Dbt multikey(&numberOfKeysRead,sizeof(numberOfKeysRead));
		Dbt multidata(&buffer,sizeof(buffer));
		multidata.set_flags(DB_DBT_USERMEM);
		multidata.set_ulen(sizeof(buffer));
		err= 0;
		while(err==0)
		{
			err= dbc->get(&multikey,&multidata,DB_NEXT|DB_MULTIPLE_KEY);
			if(err==0)
			{
				u_int32_t recno= 0;
				Dbt data;
				DbMultipleRecnoDataIterator i(multidata);
				while(err==0 && i.next(recno,data))
				{
					int actualData= *((int*)data.get_data());
					if(recno!=actualData+1)
					{
						std::cout << "Error: recno/data mismatch. " << recno << "!=" << actualData << "+1" << std::endl;
						err= -1;
					}
					numberOfKeysRead++;
				}
			} else if(err!=DB_NOTFOUND)
				std::cout << "Error: dbc->get: " << db_strerror(err) << std::endl;
		}
		if(numberOfKeysRead!=numberOfKeysToWrite)
		{
			std::cout << "Error: key count mismatch. " << numberOfKeysRead << "!=" << numberOfKeysToWrite << std::endl;
		}
		dbc->close();
	}

	db.close(0);
}

int main()
{
	test1();
	test2();
	test3();
	return (0);
}

