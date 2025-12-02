<?php  // Moodle configuration file

unset($CFG);
global $CFG;
$CFG = new stdClass();

$CFG->dbtype    = 'mariadb';
$CFG->dblibrary = 'native';
$CFG->dbhost    = 'localhost';
$CFG->dbname    = 'moodle';
$CFG->dbuser    = 'root';
$CFG->dbpass    = '';
$CFG->prefix    = 'mdl_';
$CFG->dboptions = array (
  'dbpersist' => 0,
  'dbport' => '',
  'dbsocket' => '',
  'dbcollation' => 'utf8mb4_unicode_ci',
);

$CFG->wwwroot   = 'https://6ce3d1b9c243.ngrok-free.app';  // Changed to HTTPS ngrok URL (temporary for external access)
$CFG->dataroot  = 'C:\\Users\\kiran\\Downloads\\MoodleWindowsInstaller-latest\\server\\moodledata';
$CFG->admin     = 'admin';

$CFG->directorypermissions = 0777;

$CFG->sslproxy = true;  // Enables proxy/SSL handling for ngrok
$CFG->reverseproxy = true;  // Helps with external access
$CFG->noemailever = true;  // Optional: Disables email if not set up

require_once(__DIR__ . '/lib/setup.php');

// There is no php closing tag in this file,
// it is intentional because it prevents trailing whitespace problems!
