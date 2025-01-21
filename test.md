学生表 student：id (primary key) name stu_id(unique_key)
课程表 class：id(primary key) class_name
成绩表 score：id(primary key) s_id c_id score
```sql
select * from student
where name like "%王%";
```
```sql
select s1.* from student s1 
right join score s2 on s1.id = s2.s_id
where s2.score < 60
join
select class_name from class
where id in (
    select s2.c_id as id from student s1 
    right join score s2 on s1.id = s2.s_id
    where s2.score < 60
)
```
```sql
select s1.id, avg(s2.score) from student s1 right join 
score s2 on s1.id = s2.s_id
group by s1.id
```


```java
public void main(int[] nums, int target) {
    int n = nums.length;
    Arrays.sort(nums);
    for (int i = 0; i < n; i ++) {
        int t = target - nums[i];
        int l = i + 1, r = n - 1;
        while (l < r) {
            int mid = (l + r + 1) >>> 1;
            if (nums[mid] == t) {
            sout;
            return;
            }
            if (nums[mid] > t) r = mid - 1;
            else l = mid + 1;
        }
    }
}
```