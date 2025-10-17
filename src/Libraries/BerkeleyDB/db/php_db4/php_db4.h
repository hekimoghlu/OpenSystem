/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#ifndef PHP_DB4_H
#define PHP_DB4_H

extern zend_module_entry db4_module_entry;
#define phpext_db4_ptr &db4_module_entry

#ifdef DB4_EXPORTS
#define PHP_DB4_API __declspec(dllexport)
#else
#define PHP_DB4_API
#endif

#ifdef ZTS
#include "TSRM.h"
#endif

#include "db.h"

PHP_DB4_API zend_class_entry *db_txn_ce_get(void);
PHP_DB4_API zend_class_entry *dbc_ce_get(void);
PHP_DB4_API zend_class_entry *db_env_ce_get(void);
PHP_DB4_API zend_class_entry *db_ce_get(void);
PHP_DB4_API DB_ENV *php_db4_getDbEnvFromObj(zval *z TSRMLS_DC);
PHP_DB4_API DB *php_db4_getDbFromObj(zval *z TSRMLS_DC);
PHP_DB4_API DB_TXN *php_db4_getDbTxnFromObj(zval *z TSRMLS_DC);
PHP_DB4_API DBC *php_db4_getDbcFromObj(zval *z TSRMLS_DC);
/* 
  	Declare any global variables you may need between the BEGIN
	and END macros here:     

ZEND_BEGIN_MODULE_GLOBALS(db4)
	long  global_value;
	char *global_string;
ZEND_END_MODULE_GLOBALS(db4)
*/

/* In every utility function you add that needs to use variables 
   in php_db4_globals, call TSRM_FETCH(); after declaring other 
   variables used by that function, or better yet, pass in TSRMLS_CC
   after the last function argument and declare your utility function
   with TSRMLS_DC after the last declared argument.  Always refer to
   the globals in your function as DB4_G(variable).  You are 
   encouraged to rename these macros something shorter, see
   examples in any other php module directory.
*/

#ifdef ZTS
#define DB4_G(v) TSRMG(db4_globals_id, zend_db4_globals *, v)
#else
#define DB4_G(v) (db4_globals.v)
#endif

#endif	/* PHP_DB4_H */


/*
 * Local variables:
 * tab-width: 4
 * c-basic-offset: 4
 * indent-tabs-mode: t
 * End:
 */
