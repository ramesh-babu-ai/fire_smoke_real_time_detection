#include <iostream>
#include <cstring>
#include <windows.h>
#include <string>
#include <vector>
using namespace std;
vector<string> listFiles(const char * dir);

int main()
{
	using namespace std;


	std::string src = "C:/Users/Administrator/Desktop/smoke_train_32x24";
	char const *dir = src.c_str();
	vector<string> res;
	res = listFiles(dir);
	return 0;
}

vector<string> listFiles(const char * dir)
{
	vector<string> res;
	using namespace std;

	HANDLE hFind;
	WIN32_FIND_DATA findData;
	char dirNew[100];
	strcpy(dirNew, dir);
	strcat(dirNew, "\\*.*");

	hFind = FindFirstFile(dirNew, &findData);
	do
	{
		res.push_back(findData.cFileName);
		//cout << findData.cFileName << endl;
	} while (FindNextFile(hFind, &findData));

	FindClose(hFind);
	return res;
}