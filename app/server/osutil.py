import os
import aiofiles
def create_dir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)


async def write_to_file(in_file, out_file_path):
    async with aiofiles.open(out_file_path, 'wb') as out_file:
        while content := await in_file.read(1024):  # async read file chunk
            await out_file.write(content)  # async write file chunk