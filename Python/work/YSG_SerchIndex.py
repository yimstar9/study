dataset = [5,10,18,22,35,55,75,103]
find = int(input("검색할 값 입력 :"))
success = False

for i in dataset:
    if find == i:
        print(f"{dataset}에서 {i}(은/는) {dataset.index(i)+1}번째 입니다")
        success=True
        break

if not success:
    print("찾는 값은 없습니다.")

###########################################

dataset = [5,10,18,22,35,55,75,103]
find = int(input("검색할 값 입력 :"))

if find in dataset:
        print(f"{dataset}에서 {find}은/는 {dataset.index(find)+1}번째 입니다")
else:
    print("Not found")

